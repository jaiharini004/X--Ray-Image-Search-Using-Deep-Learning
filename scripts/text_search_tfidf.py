import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load metadata
CSV_PATH = "metadata.csv"
df = pd.read_csv(CSV_PATH)

# Combine useful text fields
df["text"] = (
    df["image_name"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["dataset_name"].astype(str)+ " " +
    df["description"].astype(str)
)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

def search_images(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices].copy()
    results["score"] = similarity[top_indices]
    return results[["image_name", "category", "score"]]

# ---- Run search ----
if __name__ == "__main__":
    print("Text-Based X-ray Image Search (TF-IDF)")
    query = input("Enter search query: ")

    results = search_images(query, top_n=5)

    print("\nTop Matching Images:\n")
    print(results.to_string(index=False))
