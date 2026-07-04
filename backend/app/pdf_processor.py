import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple
import hashlib

class PDFProcessor:
    def __init__(self):
        self.font_sizes = {}
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract title, authors, and metadata using font-size heuristic"""
        doc = fitz.open(pdf_path)
        
        # Find largest font text on first page
        title_font_size = 0
        title_text = ""
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            size = span["size"]
                            text = span["text"].strip()
                            
                            # Font size heuristic for title
                            if size > 14 and len(text) > 10:
                                if size > title_font_size:
                                    title_font_size = size
                                    title_text = text
        
        # Extract authors (try to find after title)
        authors = self._extract_authors(doc, title_font_size)
        
        return {
            "title": title_text or "Untitled",
            "authors": authors,
            "page_count": len(doc)
        }
    
    def _extract_authors(self, doc, title_font_size) -> List[str]:
        """Extract author names using font size patterns"""
        authors = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Authors usually in smaller font than title
                            if span["size"] < title_font_size and span["size"] > 8:
                                text = span["text"].strip()
                                # Simple author detection
                                if "," in text or "and" in text.lower():
                                    authors.extend([a.strip() for a in text.split(",")])
        
        return authors[:5]  # Limit to 5 authors
    
    def extract_chunks(self, pdf_path: str, chunk_size: int = 500) -> List[Dict]:
        """Extract text and split into overlapping chunks"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Split into chunks with overlap
            words = text.split()
            for i in range(0, len(words), chunk_size - 100):
                chunk_text = " ".join(words[i:i + chunk_size])
                chunks.append({
                    "text": chunk_text,
                    "page": page_num + 1,
                    "chunk_id": len(chunks)
                })
        
        return chunks
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Get comprehensive PDF info"""
        doc = fitz.open(pdf_path)
        return {
            "filename": pdf_path.split("/")[-1],
            "page_count": len(doc),
            "file_size": len(doc.tobytes()),
            "metadata": doc.metadata
        }