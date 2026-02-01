import os
import csv

DATASET_DIR = r"D:\X-ray Project"
OUTPUT_CSV = "metadata.csv"
MAX_IMAGES_PER_CATEGORY = 100

# Category-wise descriptions (for TF-IDF improvement)
CATEGORY_DESCRIPTIONS = {
    "chest": "pneumonia chest xray infection lungs",
    "dental": "dental xray teeth jaw oral",
    "fractures": "bone fracture xray injury",
    "spine": "spine xray vertebra",
    "knee": "knee joint xray osteoarthritis"
}

rows = []

for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)

    if not os.path.isdir(category_path):
        continue

    if category == "scripts" or category == "venv":
        continue

    count = 0

    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(category_path):
        for image_name in files:
            if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
                description = CATEGORY_DESCRIPTIONS.get(
                    category.lower(), "xray medical image"
                )

                rows.append([
                    image_name,
                    category,
                    "Public Dataset",
                    f"{category} X-ray Dataset",
                    description
                ])

                count += 1
                if count >= MAX_IMAGES_PER_CATEGORY:
                    break

        if count >= MAX_IMAGES_PER_CATEGORY:
            break

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name",
        "category",
        "source",
        "dataset_name",
        "description"
    ])
    writer.writerows(rows)

print(f"metadata.csv created with {len(rows)} entries")
