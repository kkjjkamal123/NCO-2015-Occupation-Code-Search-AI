import faiss
import numpy as np

# Load embeddings
emb = np.load("nco_embeddings.npy").astype('float32')

# Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)

# Save index
faiss.write_index(index, "nco_index.faiss")
print(f"✅ FAISS index saved with {index.ntotal} entries")
