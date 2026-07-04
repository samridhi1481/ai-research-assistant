if (!(Test-Path README.md)) {
    @"
# 📚 AI Research Paper Intelligence System

A comprehensive AI-powered system for semantic search, summarization, keyword extraction, and named entity recognition on 50,000+ research papers from arXiv.

## Features
- Semantic Search using FAISS
- AI Summarization with BART
- Keyword Extraction with KeyBERT
- Named Entity Recognition (NER)

## Tech Stack
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS for vector search
- BART for summarization
- KeyBERT for keywords
- BERT-base-NER for entities
- Streamlit for web interface

## Installation
```bash
pip install -r requirements.txt
python src/simple_data_prep.py
python src/build_index.py
streamlit run src/app.py