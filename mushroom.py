import os
import torch
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity 
 
from diffusers import StableUnCLIPImg2ImgPipeline
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, BlipProcessor, BlipForConditionalGeneration

#required pip installs (once in terminal)
#pip install --upgrade diffusers[torch]
#pip install transformers
#pip install scikit-learn
#AND pip install umap-learn


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


#loading BLIP processor and model
model_name = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)


# ----------------------------- FUNCTIONS ---------------------------------------------------


#BLIP captioning functions
def caption_image(image_path, model=blip_model, processor=processor, max_new_tokens=30, num_beams=5):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams = num_beams
        )

    caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption 


def caption_images(image_paths, model=blip_model, processor=processor, 
                   max_new_tokens=30, num_beams=5):
    images = [Image.open(p).convert("RGB") for p in image_paths]

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )

    captions = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return captions

#ok another blip that tries to describe instead of generically, with things like cap shape 
def caption_mushroom_features(image_path, model=blip_model, processor=processor, max_new_tokens=30):
    image = Image.open(image_path).convert("RGB")
    
    # Instruction prompt
    prompt = ("Describe the details of the mushroom in terms of its cap shape, cap color, "
              "stem type, and any patterns. Be short and concise.")

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def caption_mushroom_features_batch(image_paths, max_new_tokens=30):
    captions = []
    for img in image_paths:
        caption = caption_mushroom_features(img, max_new_tokens=max_new_tokens)
        captions.append(caption)
    return captions

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

def combine_embeds2(labels, text_list):
    """
    After clustering, assign text to clusters for interpretation only.
    """
    cluster_map = {}
    for label, text in zip(labels, text_list):
        cluster_map.setdefault(label, []).append(text)
    return cluster_map

#KMEANS

def cluster(all_embeds, n=2, rs=42):
    kmeans = KMeans(n_clusters=n, random_state=rs)
    kmeans.fit(all_embeds)
    return kmeans.labels_, kmeans

#this gives ARI score (0 meeans random clustering, 1 is awesome, and negative is SAD/ WORSE than random)
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

#embeds only a portion of the images from a folder eg. first 8 of each. n is the number of images you want. 
#careful when using bc some only have 8 images max
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


#------- more functions 

from sklearn.preprocessing import normalize 
from sklearn.manifold import TSNE
import umap 

def normalize_embeds(embeds):
    return normalize(embeds)

#tsne  is another method for visualizing cluster separation, 
#umap is modern tsne 
#t-SNE
#MAY need to tweak perplex and learning rates depending on images 
#perplexity < n smaples/3
#tsne_2d = reduce_tsne(pca_embeds)  # use PCA first for speed
#plot_clusters(tsne_2d, labels)
def reduce_tsne(embeds, perplexity=15, learning_rate=200, n_components=2):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init='pca'
    )
    return tsne.fit_transform(embeds)

#umap 
#umap_2d = reduce_umap(pca_embeds)
#plot_clusters(umap_2d, labels)
def reduce_umap(embeds, n_neighbors=10, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components
    )
    return reducer.fit_transform(embeds)

from sklearn.cluster import AgglomerativeClustering, DBSCAN
#aggomerative another form of clustering 
#dbscan good for weird organic clusters, detects outliters 

#cluster_agglomerative 
#through density. high density = cluster, low dens = outlier
#labels, model = cluster_agglomerative(pca_embeds, k=2)
def cluster_agglomerative(embeds, k, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(embeds)
    return labels, model

#may need to tweak eps depending on embedding distances
def cluster_dbscan(embeds, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(embeds)
    return labels, model

#gmm gaussian mix 
#g_labels, g_probs, gmm = cluster_gmm(all_embeds, k=2)

from sklearn.mixture import GaussianMixture

def cluster_gmm(embeds, k=2, rs=42):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=rs)
    gmm.fit(embeds)
    labels = gmm.predict(embeds)
    probs = gmm.predict_proba(embeds)
    return labels, probs, gmm

#spectral cluster
#s_labels, _ = cluster_spectral(all_embeds, k=2) 
from sklearn.cluster import SpectralClustering

def cluster_spectral(embeds, k=2, rs=42):
    spec = SpectralClustering(
        n_clusters=k,
        affinity='nearest_neighbors',
        random_state=rs
    )
    labels = spec.fit_predict(embeds)
    return labels, spec

#silhouette-score 
#cluster_quality(all_embeds, k_labels)
from sklearn.metrics import silhouette_score

def cluster_quality(embeds, labels):
    return silhouette_score(embeds, labels)

#plot all clusters 
def plot_clusters(xy, labels, title="Clusters"):
    """
    xy: 2D reduced embeddings (UMAP/TSNE)
    labels: cluster assignments
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20")
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

#centroid similarity analysis 
def cluster_centroids(embeds, labels):
    centroids = []
    unique = sorted(np.unique(labels))
    for c in unique:
        centroids.append(embeds[labels == c].mean(axis=0))
    return np.vstack(centroids)

import re
#sanitizing text descriptions for CLIP 
def clean_text_description(text):
    forbidden = ["fly agaric", "fly", "agaric", 
                 "funeral bell", "galerina", "marginata"]
    t = text.lower()
    for word in forbidden:
        t = re.sub(r"\b" + word + r"\b", "", t)
    return ' '.join(t.split())


#--- plotting pairwise similarity 
#sims is the cosine similarity of the embeds 
#have the labels for row and col b the names of .. eg. the first 6 mushrooms 
def plot_similarity_heatmap(sims, row_labels=None, col_labels=None, 
                            cmap="viridis", figsize=(8, 6), title="Similarity Heatmap"):
    """
    Plots a similarity matrix as a heatmap with text annotations.
    
    sims: 2D numpy array of similarities (e.g., cosine similarity)
    row_labels: list of strings for y-axis
    col_labels: list of strings for x-axis
    cmap: colormap for heatmap
    figsize: size of the figure
    title: plot title
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Display heatmap
    im = ax.imshow(sims, cmap=cmap)

    # Add text annotations
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            ax.text(j, i, f"{sims[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)

    # Set axis ticks
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")

    # Add colorbar and title
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

