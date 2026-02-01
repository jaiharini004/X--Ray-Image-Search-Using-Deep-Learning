import pandas as pd

df = pd.read_csv("metadata.csv")

def generate_description(category):
    descriptions = {
        "chest": "pneumonia chest xray infection lungs",
        "dental": "dental xray teeth jaw oral",
        "fractures": "bone fracture xray injury",
        "spine": "spine xray vertebra",
        "knee": "knee joint xray osteoarthritis"
    }
    return descriptions.get(category, "xray image")

df["description"] = df["category"].apply(generate_description)

df.to_csv("metadata.csv", index=False)

print("Descriptions added successfully")
