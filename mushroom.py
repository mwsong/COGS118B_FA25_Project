import os
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
 
from diffusers import StableUnCLIPImg2ImgPipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

#required pip installs (once in terminal)
#pip install --upgrade diffusers[torch]
#pip install transformers
#pip install scikit-learn


#loading pipeline and encoders 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1-unclip",
    torch_dtype=torch.float16,
).to(device)

vision_encoder = pipe.image_encoder

openclip_repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
tokenizer = CLIPTokenizer.from_pretrained(openclip_repo)
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    openclip_repo,
    torch_dtype=torch.float16
).to(device)

pipe.tokenizer, pipe.text_encoder = tokenizer, text_encoder

#embedding functions
def embed_images(paths, batch_size=8):
    out, fe, enc = [], pipe.feature_extractor, pipe.image_encoder
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        px = fe(imgs, return_tensors="pt").pixel_values.to(enc.device, enc.dtype)
        with torch.no_grad():
            v = enc(px)[0]
        out.append(v)
    return torch.cat(out)

def embed_texts(prompts, batch_size=64):
    vecs = []
    for i in range(0, len(prompts), batch_size):
        toks = tokenizer(prompts[i:i + batch_size],
                         padding=True, truncation=True, max_length=77,
                         return_tensors="pt").to(text_encoder.device)
        with torch.no_grad():
            t = text_encoder(**toks).text_embeds
        vecs.append(t)
    return torch.cat(vecs)

#classify 
def classify(pipe, image, label):
    img = embed_images([image])
    txt = embed_texts(label)

    img = torch.nn.functional.normalize(img, dim=-1)
    txt = torch.nn.functional.normalize(txt, dim=-1)

    sim = (img @ txt.T).squeeze(0)

    best = sim.argmax().item()
    return label[best]

#-------------------------------------------------------------------------------------------------------- -----------------
#PCA FUNCTIONS 

#run_pca takes embeds(should be a numpy array, make sure to put in what's run from embed_whole_folder or embed_label_prompts or combine)
#n is the number of principal components, needs to be 2 to be 2d for plotting scatter plot 
#returns reduced: which is what should be inputted into plot_pca and kmeans
#returns pca: which is the actual pca model-> only to put into plotting eigenvalues to see how many principal components actually matter
def run_pca(embeds, n=2):
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(embeds)
    return reduced, pca

#to see how many principal components matter
def plot_eigenvalues(pca):
    eigenvalues = pca.explained_variance_
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker="o")
    plt.show()

def plot_pca_2d(red_embeds, labels=None, title="PCA Scatter Plot"):
    plt.figure(figsize=(6,6))

    if labels is None:
        plt.scatter(red_embeds[:,0], red_embeds[:,1])
    else:
        for lab in np.unique(labels):
            idx = [i for i,x in enumerate(labels) if x==lab]
            plt.scatter(red_embeds[idx,0], red_embeds[idx,1], label=str(lab))
        plt.legend()
    plt.show()

#COMBINE EMBEDS

#combines and repeats the text as equally as possible bc if img embed and text embed different number of rows it can't properly 
#attach one text to one image row. so eg. if there's 8 images and 2 text rows, then it'll do the first text row for the first 
#4 images, then the second one for the next 4 so make sure we order things correctly. eg. 'beechwood' should attach to the
#beechwood images 
def combine_embeds(img_embeds, text_embeds, repeat_text=True):
    N_images = img_embeds.shape[0]
    N_texts  = text_embeds.shape[0]

    if N_images != N_texts:
        if repeat_text:
            if N_texts == 1:
                text_embeds = np.tile(text_embeds, (N_images, 1))
            else:
                reps = int(np.ceil(N_images / N_texts))
                text_embeds = np.tile(text_embeds, (reps, 1))[:N_images]
        else:
            raise ValueError("Mismatch")

    return np.concatenate([img_embeds, text_embeds], axis=1)

#KMEANS

def cluster(all_embeds, n=2, rs=42):
    kmeans = KMeans(n_clusters=n, random_state=rs)
    kmeans.fit(all_embeds)
    return kmeans.labels_, kmeans

def cluster_accuracy(true_labels, cluster_labels):
    return adjusted_rand_score(true_labels, cluster_labels)

#EMBEDDING MULTIPLE (IMAGES/TEXT)

#embeds a whole folder of images (need to specify root as 'mushroom')
def embed_whole_folder(folder_path, bs=8):
    img_files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.endswith(".png")]
    embeds = embed_images(img_files, batch_size=bs)
    return embeds.cpu().numpy()

#embeds a whole array of labels 
def embed_label_prompts(labels, bs=64):
    embeds = embed_texts(labels, batch_size=bs)
    return embeds.cpu().numpy()

#embeds only a portion of the images from a folder eg. first 8 of each. n is the number of images you want
def embed_first_n_images(folder_path, n=1, bs=8):
    img_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    ]

    img_files.sort()
    img_files = img_files[:n]

    embeds = embed_images(img_files, batch_size=bs)
    return embeds.cpu().numpy()

#classify images (multiple)
def classify_images(pipe, image_paths, labels):
    return [classify(pipe, img, labels) for img in image_paths]
