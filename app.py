"""
Moteur de Recherche d'Images Sémantique — Démo Streamlit
CLIP (ViT-B/32) + FAISS, sur un corpus COCO pré-encodé.

Lancement local :
    streamlit run app.py
"""

import json
import os

import faiss
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

EXPORT_DIR = "clip_search_export"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

st.set_page_config(page_title="Recherche d'images sémantique", layout="wide")


# ----------------------------------------------------------------
# Chargement des ressources (mis en cache : une seule fois par session serveur)
# ----------------------------------------------------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor, device


@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(os.path.join(EXPORT_DIR, "faiss_index.bin"))
    with open(os.path.join(EXPORT_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return index, metadata


@torch.no_grad()
def encode_query(text: str, model, processor, device) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    embed = model.get_text_features(**inputs)
    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
    return embed.cpu().numpy().astype("float32")


def search(query: str, k: int, model, processor, device, index, metadata):
    query_embedding = encode_query(query, model, processor, device)
    scores, indices = index.search(query_embedding, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        meta = metadata[idx]
        results.append({**meta, "score": float(score)})
    return results


# ----------------------------------------------------------------
# Interface
# ----------------------------------------------------------------
st.title("🔍 Moteur de Recherche d'Images Sémantique")
st.caption(
    "Recherche zero-shot par similarité CLIP + FAISS sur un corpus de 10 000 images COCO. "
    "Aucun mot-clé exact requis : décrivez la scène en langage naturel."
)

model, processor, device = load_clip()
index, metadata = load_index_and_metadata()

col_query, col_k = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Décrivez l'image recherchée",
        placeholder="ex : a dog running on the beach",
    )
with col_k:
    k = st.slider("Nombre de résultats", min_value=1, max_value=10, value=5)

st.markdown("**Exemples rapides :**")
example_cols = st.columns(4)
examples = [
    "a dog running on the beach",
    "a group of people playing soccer",
    "a man wearing blue",
    "a plate of food on a table",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex):
        query = ex

if query:
    with st.spinner("Recherche en cours..."):
        results = search(query, k, model, processor, device, index, metadata)

    st.subheader(f"Résultats pour : « {query} »")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        img_path = os.path.join(EXPORT_DIR, r["image_path"])
        image = Image.open(img_path)
        col.image(image, use_container_width=True)
        col.markdown(f"**Score : {r['score']:.3f}**")
        col.caption(r["caption"])
else:
    st.info("Entrez une requête ou cliquez sur un exemple pour lancer une recherche.")

with st.expander("À propos du modèle"):
    st.markdown(
        """
        - **Encodeur** : CLIP ViT-B/32 (openai/clip-vit-base-patch32), gelé, zero-shot
        - **Indexation** : FAISS `IndexFlatIP` (recherche exacte par produit scalaire)
        - **Corpus** : 10 000 images COCO (Karpathy split), embeddings pré-calculés
        - **Métrique** : similarité cosinus (embeddings normalisés en norme L2)
        """
    )
