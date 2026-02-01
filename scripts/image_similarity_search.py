import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# Files
EMBEDDINGS_FILE = "image_embeddings.npy"
IMAGE_NAMES_FILE = "image_names.npy"

# Load embeddings
embeddings = np.load(EMBEDDINGS_FILE)
image_names = np.load(IMAGE_NAMES_FILE)

# Load pretrained ResNet50 (same as embedding step)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image preprocessing (must match embedding step)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze().numpy()

def search_similar_images(query_image_path, top_n=5):
    query_emb = extract_embedding(query_image_path)
    similarities = cosine_similarity(
        query_emb.reshape(1, -1),
        embeddings
    ).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "image_name": image_names[idx],
            "similarity_score": similarities[idx]
        })

    return results

# ---- Run search ----
if __name__ == "__main__":
    print("Image-Based X-ray Similarity Search (ResNet-50)")
    print(f"Loaded {len(image_names)} embeddings\n")
    
    query_path = input("Enter path to query X-ray image: ").strip().strip('"').strip("'")

    try:
        results = search_similar_images(query_path, top_n=5)

        print("\nTop Similar Images:\n")
        for r in results:
            print(f"{r['image_name']}  |  score: {r['similarity_score']:.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nExample: D:\\X-ray Project\\chest\\test\\PNEUMONIA\\PNEUMONIA(3505).jpg")
    except Exception as e:
        print(f"Error: {e}")
