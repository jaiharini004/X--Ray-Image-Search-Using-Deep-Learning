# X-Ray Image Search Interface

## Objective
The objective of this project is to build a prototype system that enables users to search X-ray images using either a **text query** or an **image query**.  
The system retrieves the most relevant or visually similar X-ray images from a curated medical image dataset.

---

## Dataset Collection
- Minimum of **500 X-ray images**
- Images collected from **at least 5 public medical datasets**
- Multiple categories included:
  - Chest
  - Spine
  - Dental
  - Fracture
  - Knee / Joint
- A metadata CSV file is maintained with:
  - image_name
  - category
  - dataset/source information

---

## Text-Based Search
- Users can enter a **text query** (e.g., “chest x-ray”, “knee fracture”).
- Metadata text is converted into numerical vectors using **TF–IDF**.
- **Cosine similarity** is used to rank images.
- The system returns the **top 5–10 most relevant images**.

---

## Image-Based Search (Reverse Image Search)
- Users can upload an X-ray image as a query.
- Pretrained deep learning models are used for feature extraction:
  - ResNet
  - DenseNet
  - Vision Transformer (ViT)
  - CLIP
- Feature embeddings of the query image are compared with dataset embeddings.
- **Cosine similarity** is used to retrieve the **top 5 most similar X-ray images**.

---

## User Interface
- Supports:
  - Text input for search
  - Image upload for reverse search
  - Grid-based result display
- Handles invalid inputs and missing results gracefully.
- Implemented as a lightweight prototype UI.

---

## Project Structure
X-ray Project/
├── chest/
├── dental/
├── fractures/
├── knee/
├── spine/
├── scripts/
│ ├── create_metadata_csv.py
│ ├── text_search_tfidf.py
│ └── image_search.py
├── metadata.csv
├── requirements.txt
└── README.md


---

## How to Run

### 1. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python scripts/text_search_tfidf.py
streamlit run app.py