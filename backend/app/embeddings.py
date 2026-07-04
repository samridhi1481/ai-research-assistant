# Option 1: Try this import
from builtins import ImportError


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Option 2: Fallback to older import
    import sentence_transformers
    SentenceTransformer = sentence_transformers.SentenceTransformer

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.chunks = []
        
    # ... rest of the class remains the same


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.chunks = []
        
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to index"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        self.chunks.extend(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "similarity": float(dist),
                    "index": int(idx)
                })
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index and chunks"""
        if self.index:
            faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.chunks", "wb") as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self, path: str):
        """Load FAISS index and chunks"""
        self.index = faiss.read_index(f"{path}.faiss")
        with os.open(f"{path}.chunks", "rb") as f:
            self.chunks = pickle.load(f)