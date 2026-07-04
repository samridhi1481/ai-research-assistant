"""
AI Research Paper Search Engine
Combines: Semantic Search + Summarization + Keyword Extraction
"""

import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
import warnings
warnings.filterwarnings('ignore')

class PaperSearchEngine:
    def __init__(self, data_path=None, embedding_path=None):
        """Initialize the search engine with data and models"""
        
        # Set paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if data_path is None:
            data_path = os.path.join(self.base_dir, "data", "cleaned_arxiv_papers.csv")
        if embedding_path is None:
            embedding_path = os.path.join(self.base_dir, "data", "arxiv_embeddings.npy")
        
        print("📚 Loading data...")
        self.df = pd.read_csv(data_path)
        
        print("🧠 Loading embeddings...")
        self.embeddings = np.load(embedding_path)
        
        print("🤖 Loading model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("🔍 Building FAISS index...")
        self.index = self._build_index()
        
        print("📝 Loading summarizer...")
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU
        )
        
        print("🏷️ Loading KeyBERT...")
        self.kw_model = KeyBERT(model=self.model)
        
        print("✅ Search Engine ready!")
    
    def _build_index(self):
        """Build FAISS index for fast search"""
        faiss_embeddings = self.embeddings.copy()
        faiss.normalize_L2(faiss_embeddings)
        index = faiss.IndexFlatIP(384)
        index.add(faiss_embeddings)
        return index
    
    def search(self, query, k=5):
        """Search for similar papers"""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "score": float(score),
                "title": self.df.iloc[idx]['title'],
                "abstract": self.df.iloc[idx]['abstract'],
                "paper_text": self.df.iloc[idx].get('paper_text', self.df.iloc[idx]['abstract'])
            })
        return results
    
    def summarize(self, text, max_length=120, min_length=40):
        """Summarize text using BART"""
        if len(text) < 100:
            return "Text too short to summarize."
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error summarizing: {str(e)}"
    
    def extract_keywords(self, text, top_n=5):
        """Extract keywords using KeyBERT"""
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=top_n
            )
            return keywords
        except Exception as e:
            return []
    
    def full_report(self, query, k=3):
        """Get complete report: search + summary + keywords"""
        results = self.search(query, k)
        
        for result in results:
            result["summary"] = self.summarize(result["abstract"])
            result["keywords"] = self.extract_keywords(result["paper_text"])
        
        return results
    
    def save_index(self, path="data/paper_faiss.index"):
        """Save FAISS index to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"✅ Index saved to {path}")
    
    def load_index(self, path="data/paper_faiss.index"):
        """Load FAISS index from disk"""
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            print(f"✅ Index loaded from {path}")
            return True
        return False