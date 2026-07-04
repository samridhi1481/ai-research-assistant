"""
Data Preparation Script
Loads and cleans the ArXiv dataset
"""

from datasets import load_dataset
import pandas as pd
import os

def prepare_data(n_samples=50000, save_path="data/cleaned_arxiv_papers.csv"):
    """
    Download and prepare the ArXiv dataset
    
    Args:
        n_samples: Number of papers to keep
        save_path: Where to save the cleaned CSV
    """
    print("📥 Loading dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    print(f"📊 Original dataset size: {len(df)}")
    
    # Keep only needed columns
    df = df[['title', 'abstract']]
    
    # Take only n_samples
    df = df.head(n_samples)
    
    # Clean data
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    
    # Create combined text
    df["paper_text"] = df["title"] + " " + df["abstract"]
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
    df["paper_text"] = df["paper_text"].str.strip()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Saved cleaned dataset to {save_path}")
    print(f"📊 Dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    prepare_data()