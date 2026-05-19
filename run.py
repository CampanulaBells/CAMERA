import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(dataset_name, data_dir="./dataset_cleaned"):
    data = torch.load(f"{data_dir}/{dataset_name}.pt")
    with open(f"{data_dir}/emb_openai/embeddings_{dataset_name}.pkl", "rb") as f:
        res_dict = pickle.load(f)
    embeddings = torch.tensor(np.array(res_dict["embeddings"]), dtype=torch.float32)
    data.x = embeddings
    return data.to(device)

def build_adj(data):
    edge_index = data.edge_index.to(device)
    row, col = edge_index
    adj = SparseTensor(
        row=row,
        col=col,
        sparse_sizes=(data.x.size(0), data.x.size(0))
    )
    deg = adj.sum(dim=1).clamp(min=1).unsqueeze(1)
    return adj, deg

def compute_context(x, adj, deg):
    return adj.matmul(x) / deg

class GraphExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)

    def forward(self, h, agg):
        pred = self.mlp(agg.detach())
        residual = h - pred
        loss = residual.pow(2).sum(dim=1).mean()
        return residual, loss

class NormalExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)

    def forward(self, h):
        mean_h = h.mean(dim=0, keepdim=True)
        residual = h - self.mlp(mean_h)
        loss = residual.pow(2).sum(dim=1).mean()
        return residual, loss

class SemanticExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, dim)

    def forward(self, h):
        z = F.relu(self.encoder(h))
        recon = self.decoder(z)
        residual = h - recon
        loss = F.mse_loss(recon, h)
        return residual, loss

class GatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, 3)

    def forward(self, h, context):
        z = torch.cat([h, context], dim=1)
        return F.softmax(self.fc(z), dim=1)

class MOELayer(nn.Module):
    def __init__(self, dim, sem_hidden):
        super().__init__()
        self.graph = GraphExpert(dim)
        self.normal = NormalExpert(dim)
        self.semantic = SemanticExpert(dim, sem_hidden)
        self.gating = GatingNetwork(dim)

    def forward(self, h, context):
        r_g, loss_g = self.graph(h, context)
        r_n, loss_n = self.normal(h)
        r_s, loss_s = self.semantic(h)
        expert_loss = loss_g + loss_n + loss_s
        residuals = torch.stack([r_g, r_n, r_s], dim=1)
        weight = self.gating(h.detach(), context.detach())
        fused = (weight.unsqueeze(-1) * residuals).sum(dim=1)
        return fused, expert_loss, weight, residuals

class StackedMOE(nn.Module):
    def __init__(self, dim, sem_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [MOELayer(dim, sem_hidden) for _ in range(num_layers)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj, deg):
        h = x
        expert_loss_total = 0.0
        weights = []
        for layer in self.layers:
            h_old = h
            context = compute_context(h, adj, deg)
            h, expert_loss, weight, residuals = layer(h, context)
            h = h + h_old
            expert_loss_total = expert_loss_total + expert_loss
            weights.append(weight)
        score = self.sigmoid(h.norm(dim=1))
        return score, expert_loss_total, weights, residuals

def run_single_experiment(
    data,
    lr,
    alpha,
    beta,
    epochs,
    sem_hidden=64,
    num_layers=2,
    supervised=True
):
    x = data.x.to(device)
    y = data.y.to(device).float()
    adj, deg = build_adj(data)

    model = StackedMOE(
        dim=x.size(1),
        sem_hidden=sem_hidden,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_dummy = torch.zeros_like(y)

    log = []

    for epoch in range(epochs):
        model.train()
        score, expert_loss, weights, residuals = model(x, adj, deg)

        entropy_loss = 0.0
        for weight in weights:
            entropy_loss = entropy_loss - (
                weight * (weight + 1e-8).log()
            ).sum(dim=1).mean()

        if supervised:
            supervised_loss = nn.BCELoss()(score, y_dummy)
        else:
            supervised_loss = torch.tensor(0.0, device=device)

        total_loss = expert_loss + alpha * entropy_loss + beta * supervised_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(
                y.cpu().numpy(),
                score.cpu().numpy()
            )
            prc = average_precision_score(
                y.cpu().numpy(),
                score.cpu().numpy()
            )



        log.append(
            {
                "epoch": epoch,
                "loss": float(total_loss.item()),
                "expert_loss": float(expert_loss.item()),
                "entropy_loss": float(entropy_loss.item()),
                "auc": float(auc),
                "prc": float(prc),
                "lr": lr,
                "alpha": alpha,
                "beta": beta,
            }
        )

    return log, log[-1]["auc"], log[-1]["prc"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="instagram")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    data = load_data(args.dataset_name)
    log, final_auc, final_prc = run_single_experiment(
        data=data,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        epochs=args.epochs
    )
    print(args.dataset_name,   final_auc,   final_prc)
