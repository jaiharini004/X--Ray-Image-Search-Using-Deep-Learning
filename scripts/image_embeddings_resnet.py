import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

DATASET_DIR = r"D:\X-ray Project"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
EMBEDDINGS_FILE = "image_embeddings.npy"
IMAGE_NAMES_FILE = "image_names.npy"

MAX_IMAGES = 500  # 🔴 FIXED LIMIT

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

embeddings = []
image_names = []

def extract_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze().numpy()

processed = 0
stop_processing = False

for category in os.listdir(DATASET_DIR):
    if stop_processing:
        break
    
    category_path = os.path.join(DATASET_DIR, category)

    if not os.path.isdir(category_path):
        continue
    if category in ["scripts", "venv"]:
        continue

    for root, dirs, files in os.walk(category_path):
        if stop_processing:
            break
        
        for file in files:
            if file.lower().endswith(IMAGE_EXTS):
                path = os.path.join(root, file)
                try:
                    emb = extract_embedding(path)
                    embeddings.append(emb)
                    image_names.append(file)
                    processed += 1

                    if processed % 100 == 0:
                        print(f"Processed {processed} images...")

                    if processed >= MAX_IMAGES:
                        print(f"Reached {MAX_IMAGES} images. Stopping.")
                        stop_processing = True
                        break
                except Exception as e:
                    print(f"Skipping (error): {path}")

np.save(EMBEDDINGS_FILE, np.array(embeddings))
np.save(IMAGE_NAMES_FILE, np.array(image_names))

print(f"Extracted embeddings for {len(image_names)} images")