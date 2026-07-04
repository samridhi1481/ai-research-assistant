import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from keybert import KeyBERT
import time

# Page config
st.set_page_config(
    page_title="AI Research Paper Intelligence System",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .entity-ORG { background: #ffd700; color: #333; }
    .entity-PER { background: #87ceeb; color: #333; }
    .entity-LOC { background: #98fb98; color: #333; }
    .entity-MISC { background: #dda0dd; color: #333; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📚 AI Research Paper Intelligence System</h1><p>Semantic Search + Summarization + Keyword Extraction + NER</p></div>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    print("📚 Loading data...")
    df = pd.read_csv("data/cleaned_arxiv_papers.csv")
    
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("🔍 Loading FAISS index...")
    index = faiss.read_index("data/paper_faiss.index")
    
    print("📝 Loading summarizer...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )
    
    print("🏷️ Loading KeyBERT...")
    kw_model = KeyBERT(model=model)
    
    print("🔍 Loading NER model...")
    ner_model = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1
    )
    
    return df, model, index, summarizer, kw_model, ner_model

# Load everything
df, model, index, summarizer, kw_model, ner_model = load_models()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of results", min_value=3, max_value=20, value=5)
    
    st.markdown("---")
    st.header("🎯 Features")
    show_summary = st.checkbox("Show Summarization", value=True)
    show_keywords = st.checkbox("Show Keywords", value=True)
    show_ner = st.checkbox("Show NER (Named Entity Recognition)", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.markdown(f"- Papers indexed: {len(df):,}")
    st.markdown("- Embedding model: MiniLM-L6-v2")
    st.markdown("- Similarity: Cosine")
    st.markdown("- NER Model: BERT-base-NER")

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""

# Quick Examples
st.markdown("### 🎯 Quick Examples:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧠 Deep Learning"):
        st.session_state.query = "deep learning for medical image analysis"
        st.rerun()

with col2:
    if st.button("🤖 Transformers"):
        st.session_state.query = "transformer models for natural language processing"
        st.rerun()

with col3:
    if st.button("🎮 Reinforcement"):
        st.session_state.query = "reinforcement learning in robotics"
        st.rerun()

col4, col5, col6 = st.columns(3)

with col4:
    if st.button("🖼️ GANs"):
        st.session_state.query = "generative adversarial networks for image generation"
        st.rerun()

with col5:
    if st.button("🔍 Explainable AI"):
        st.session_state.query = "explainable AI and interpretability"
        st.rerun()

with col6:
    if st.button("📊 NLP"):
        st.session_state.query = "natural language processing techniques"
        st.rerun()

# Main search input
query = st.text_input(
    "🔍 Enter your research query",
    value=st.session_state.query,
    placeholder="e.g., deep learning for medical image analysis"
)

if query != st.session_state.query:
    st.session_state.query = query

search_clicked = st.button("🔍 Search", use_container_width=True)

# Search logic
if search_clicked and st.session_state.query:
    with st.spinner("🔍 Searching and analyzing papers..."):
        start_time = time.time()
        search_query = st.session_state.query
        
        # Encode query
        query_embedding = model.encode([search_query])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        D, I = index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            result = {
                "score": float(score),
                "title": df.iloc[idx]['title'],
                "abstract": df.iloc[idx]['abstract'],
                "paper_text": df.iloc[idx]['paper_text']
            }
            results.append(result)
        
        end_time = time.time()
        
        # Process results
        for result in results:
            if show_summary:
                try:
                    summary = summarizer(result["abstract"], max_length=120, min_length=40)
                    result["summary"] = summary[0]['summary_text']
                except:
                    result["summary"] = "Could not generate summary."
            else:
                result["summary"] = None
            
            if show_keywords:
                try:
                    keywords = kw_model.extract_keywords(
                        result["paper_text"],
                        keyphrase_ngram_range=(1, 3),
                        stop_words="english",
                        top_n=5
                    )
                    result["keywords"] = keywords
                except:
                    result["keywords"] = []
            else:
                result["keywords"] = []
            
            if show_ner:
                try:
                    entities = ner_model(result["abstract"][:1000])
                    result["entities"] = entities
                except:
                    result["entities"] = []
            else:
                result["entities"] = []
        
        # Display results
        st.markdown(f"### 📊 Found {len(results)} results in {end_time - start_time:.2f} seconds")
        st.markdown(f"**Query:** '{search_query}'")
        st.markdown("---")
        
        for i, result in enumerate(results):
            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                
                # Title and score
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"#### {i+1}. {result['title']}")
                with col2:
                    st.markdown(f"**Score:** {result['score']:.4f}")
                
                # Summary
                if result.get("summary"):
                    with st.expander("📝 Summary", expanded=True):
                        st.write(result["summary"])
                
                # Keywords
                if result.get("keywords"):
                    with st.expander("🏷️ Keywords", expanded=True):
                        for keyword, score in result["keywords"]:
                            st.markdown(f"- **{keyword}** ({score:.2f})")
                
                # NER Entities
                if result.get("entities"):
                    with st.expander("🔍 Named Entities (NER)", expanded=True):
                        entity_colors = {
                            "ORG": "ORG",
                            "PER": "PER",
                            "LOC": "LOC",
                            "MISC": "MISC"
                        }
                        for entity in result["entities"]:
                            label = entity['entity_group']
                            word = entity['word']
                            color_class = entity_colors.get(label, "")
                            st.markdown(f'<span class="entity-tag entity-{color_class}">{label}: {word}</span>', unsafe_allow_html=True)
                
                # Full abstract
                with st.expander("📄 Full Abstract"):
                    st.write(result["abstract"])
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

elif search_clicked and not st.session_state.query:
    st.warning("⚠️ Please enter a search query!")

elif st.session_state.query and not search_clicked:
    st.info(f"👆 Click 'Search' to find papers for: **{st.session_state.query}**")

else:
    st.info("💡 Enter a query and click 'Search' to get started!")