"""
Build Embeddings and FAISS Index
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def build_embeddings_and_index(data_path="data/cleaned_arxiv_papers.csv",
                              embedding_path="data/arxiv_embeddings.npy",
                              index_path="data/paper_faiss.index"):
    """
    Generate embeddings and build FAISS index
    """
    print("📚 Loading cleaned data...")
    df = pd.read_csv(data_path)
    
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate or load embeddings
    if os.path.exists(embedding_path):
        print("🔄 Loading existing embeddings...")
        embeddings = np.load(embedding_path)
    else:
        print("🔄 Generating embeddings... (this will take 20-30 minutes)")
        embeddings = model.encode(
            df["paper_text"].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, embeddings)
        print(f"✅ Embeddings saved to {embedding_path}")
    
    print(f"📊 Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    if os.path.exists(index_path):
        print("🔄 Loading existing FAISS index...")
        index = faiss.read_index(index_path)
    else:
        print("🔍 Building FAISS index...")
        faiss_embeddings = embeddings.copy()
        faiss.normalize_L2(faiss_embeddings)
        index = faiss.IndexFlatIP(384)
        index.add(faiss_embeddings)
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        print(f"✅ FAISS index saved to {index_path}")
    
    print(f"📊 Papers indexed: {index.ntotal}")
    
    return embeddings, index

if __name__ == "__main__":
    build_embeddings_and_index()