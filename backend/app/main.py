from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from .pdf_processor import PDFProcessor
from .embeddings import EmbeddingManager
from .rag_pipeline import RAGPipeline
from .models import SessionLocal, Paper

app = FastAPI(title="AI Research Assistant")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
embedding_manager = EmbeddingManager()
rag_pipeline = RAGPipeline()

# Models
class QuestionRequest(BaseModel):
    question: str
    paper_id: int

class SearchQuery(BaseModel):
    query: str
    k: int = 5

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    # Save file
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF
    metadata = pdf_processor.extract_metadata(file_path)
    chunks = pdf_processor.extract_chunks(file_path)
    
    # Add to index
    embedding_manager.add_documents(chunks)
    
    # Save to database
    db = SessionLocal()
    paper = Paper(
        title=metadata["title"],
        authors=", ".join(metadata["authors"]),
        filename=file.filename,
        page_count=metadata["page_count"]
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    db.close()
    
    # Save index
    embedding_manager.save_index(f"data/index_{paper.id}")
    
    return {
        "paper_id": paper.id,
        "metadata": metadata,
        "chunks": len(chunks)
    }

@app.post("/search")
async def search_papers(query: SearchQuery):
    """Search for papers using query"""
    results = embedding_manager.search(query.query, query.k)
    return results

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about a paper"""
    # Get paper chunks
    embedding_manager.load_index(f"data/index_{request.paper_id}")
    
    # Get relevant chunks
    relevant = embedding_manager.search(request.question, k=3)
    context = " ".join([r["chunk"]["text"] for r in relevant])
    
    # Generate answer
    answer = rag_pipeline.answer_question(context, request.question)
    
    return {
        "question": request.question,
        "answer": answer,
        "sources": relevant
    }

@app.post("/analyze/{paper_id}")
async def analyze_paper(paper_id: int):
    """Generate comprehensive analysis of a paper"""
    embedding_manager.load_index(f"data/index_{paper_id}")
    
    # Get all chunks
    chunks = embedding_manager.chunks
    full_text = " ".join([chunk["text"] for chunk in chunks])
    
    analysis = rag_pipeline.generate_analysis(full_text)
    
    return analysis

@app.get("/papers")
async def get_papers():
    """Get all uploaded papers"""
    db = SessionLocal()
    papers = db.query(Paper).all()
    db.close()
    return papers

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)