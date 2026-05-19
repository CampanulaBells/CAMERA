

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import tiktoken
import pickle
from openai import OpenAI

client = OpenAI()

model = "text-embedding-3-small"


max_tokens = 8192 
batch_size = 1024

enc = tiktoken.encoding_for_model(model)

for dataset_name in ["reddit", "instagram", "amazon", "yelpchi"]:
    data = torch.load(f'./dataset_cleaned/{dataset_name}.pt')
    
    texts = data.raw_texts
    truncated_texts = []
    for text in texts:
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_texts.append(text)

    embeddings = []
    for start in range(0, len(truncated_texts), batch_size):
        end = start + batch_size
        batch = truncated_texts[start:end]

        print(f"Processing batch {start} to {end}")

        resp = client.embeddings.create(
            model=model,
            input=batch
        )

        embeddings.extend([item.embedding for item in resp.data])

    with open(f"./dataset_cleaned/emb_openai/embeddings_{dataset_name}.pkl", "wb") as f:
        pickle.dump({
            "texts": truncated_texts,
            "embeddings": embeddings
        }, f)

    print(f"Embeddings saved for {dataset_name}")


