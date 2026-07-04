from datasets import load_dataset
import pandas as pd
import os

def prepare_data():
    print(" Loading dataset from Hugging Face...")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    
    print(" Converting to DataFrame...")
    df = pd.DataFrame(dataset)
    df = df[['title', 'abstract']]
    df = df.head(50000)
    
    print(" Cleaning data...")
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    df["paper_text"] = df["title"] + " " + df["abstract"]
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
    df["paper_text"] = df["paper_text"].str.strip()
    
    print(" Saving to CSV...")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_arxiv_papers.csv", index=False)
    
    print(f"Saved {len(df)} papers to data/cleaned_arxiv_papers.csv")
    return df

if __name__ == "__main__":
    prepare_data()