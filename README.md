# CAMERA: Adapting to Semantic Camouflage in Unsupervised Text Attributed Graph Fraud Detection

This repository contains the official implementation of CAMERA, an unsupervised graph fraud detection framework designed to adapt to semantic camouflage in text attributed graphs. CAMERA employs a mixture of experts architecture to disentangle structural, statistical, and semantic signals, enabling robust anomaly detection under distribution shift without relying on labeled data.

## Environment

Python 3.11
- torch==2.5.1+cu121  
- torch-sparse==0.6.18+pt25cu121  
- numpy==1.26.4  
- scikit-learn==1.6.1  
- pickle
- sentence-transformers==3.3.1  
- torch-geometric==2.6.1 

## Usage

### Step 0: Set up the dataset

You can download the dataset points from:
https://drive.google.com/file/d/1z75C7Cul7z4SYNhyZn6Jit7cY18xFbEA/view?usp=sharing  

To make experimentation easier and reduce reproduction costs, we also provide text encodings generated using the OpenAI encoding model, available at: https://drive.google.com/file/d/1PqaIm_m3DVZbjdoe8NIuMupik3rtJO42/view?usp=sharing

### Step 1 Text Encoding

First, generate text embeddings for each dataset.

``` 
python text_encoder.py  
```



### Step 2 Run CAMERA

After text embeddings are prepared, run the main training and evaluation script with dataset specific hyperparameters.

Below are the hyperparameter settings used in our experiments.

#### Reddit
```
python run.py --dataset_name reddit --epochs 1200 --alpha 5.0 --beta 0.1 --lr 1e-3
```
#### Instagram
```
python run.py --dataset_name instagram --epochs 15 --alpha 10.0 --beta 10.0 --lr 5e-5
```
#### Amazon
```
python run.py --dataset_name amazon --epochs 15 --alpha 0.1 --beta 1.0 --lr 5e-5
```
#### YelpChi
```
python run.py --dataset_name yelpchi --epochs 450 --alpha 0.1 --beta 10.0 --lr 1e-3
```
