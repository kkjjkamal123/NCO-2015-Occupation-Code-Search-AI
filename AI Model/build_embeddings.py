from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("nco_clean.csv")

# Use a small, fast, high-quality model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the 'text' column
print("🔄 Generating embeddings...")
embeddings = model.encode(
    df['text'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# Save embeddings and updated dataset
np.save("nco_embeddings.npy", embeddings)
df.to_csv("nco_with_embeddings.csv", index=False)

print(f"✅ Saved embeddings to nco_embeddings.npy and updated dataset to nco_with_embeddings.csv")
