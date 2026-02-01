import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="X-Ray Image Search", layout="wide")

st.title("🩻 X-Ray Image Search Interface")
st.write("Search X-ray images using **text queries** or **image upload**")

# ----------------------------
# Load metadata for text search
# ----------------------------
df = pd.read_csv("metadata.csv")

df["text"] = (
    df["image_name"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["dataset_name"].astype(str) + " " +
    df["description"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

def text_search(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = similarity.argsort()[-top_n:][::-1]

    results = df.iloc[top_idx].copy()
    results["score"] = similarity[top_idx]
    return results[["image_name", "category", "score"]]

# ----------------------------
# Load image embeddings
# ----------------------------
embeddings = np.load("image_embeddings.npy")
image_names = np.load("image_names.npy")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(img):
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze().numpy()

# ----------------------------
# UI Layout
# ----------------------------
tab1, tab2 = st.tabs(["🔎 Text-Based Search", "📷 Image-Based Search"])

# -------- TEXT SEARCH TAB --------
with tab1:
    st.subheader("Text-Based X-Ray Search")

    query = st.text_input("Enter search query (e.g., chest xray pneumonia)")

    if st.button("Search Text"):
        if query.strip() == "":
            st.warning("Please enter a search query.")
        else:
            results = text_search(query)
            st.dataframe(results, use_container_width=True)

# -------- IMAGE SEARCH TAB --------
with tab2:
    st.subheader("Image-Based X-Ray Search")

    uploaded_file = st.file_uploader(
        "Upload an X-ray image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Query Image", width=300)

        if st.button("Find Similar Images"):
            query_emb = extract_embedding(img)
            sims = cosine_similarity(
                query_emb.reshape(1, -1), embeddings
            ).flatten()

            top_idx = sims.argsort()[-5:][::-1]

            st.subheader("Top Similar Images")
            for i in top_idx:
                st.write(f"**{image_names[i]}** — similarity: `{sims[i]:.4f}`")
