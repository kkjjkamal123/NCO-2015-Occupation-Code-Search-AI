from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from textwrap import shorten

# Load model & data
print("🔄 Loading model and index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("nco_index.faiss")
df = pd.read_csv("nco_with_embeddings.csv")

def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    return txt.replace("Title ", "").strip()

def search_jobs(query, k=5):
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(q_emb, k)

    results = df.iloc[I[0]].copy()
    results["similarity"] = [1 / (1 + d) for d in D[0]]  # ✅ proper similarity
    results["title_desc"] = results["title_desc"].apply(clean_text)
    results["description_desc"] = results["description_desc"].apply(clean_text)

    return results[["code", "title_desc", "description_desc", "similarity"]]

if __name__ == "__main__":
    while True:
        query = input("\nEnter a job description (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = search_jobs(query, k=5)

        print("\n📌 Top Matches:\n")
        for _, row in results.iterrows():
            print(f"🔹 Code: {row['code']}")
            print(f"   Title: {row['title_desc']}")
            print(f"   Description: {shorten(str(row['description_desc']), width=150, placeholder='...')}")
            print(f"   Similarity Score: {row['similarity']:.3f}")
            print("-" * 90)
