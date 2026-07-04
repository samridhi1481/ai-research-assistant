from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict

class RAGPipeline:
    def __init__(self, model_name="google/flan-t5-base"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.max_length = 512
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"✅ RAG Pipeline loaded with {model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to a simpler model
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.device = "cpu"
            self.model.to(self.device)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 400) -> List[str]:
        """Split text into chunks for map-reduce"""
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) < max_chunk_size:
                current_chunk += sent + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_chunk_summary(self, chunk: str, prompt: str) -> str:
        """Generate summary for a single chunk"""
        chunk_text = chunk[:350]
        full_prompt = f"{prompt}\n\nText: {chunk_text}\n\nAnswer:"
        
        inputs = self.tokenizer.encode(
            full_prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_length=150,
            min_length=30,
            temperature=0.3,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def answer_question(self, context: str, question: str) -> str:
        """Answer question using map-reduce"""
        if not context or len(context) < 50:
            return "Not enough context to answer. Please upload a paper first."
        
        chunks = self._chunk_text(context)
        
        if not chunks:
            return "Could not process the text."
        
        # MAP: Generate answers for each chunk
        partial_answers = []
        for chunk in chunks[:5]:  # Limit to first 5 chunks
            prompt = f"Answer this question based on the text: '{question}'"
            answer = self._generate_chunk_summary(chunk, prompt)
            if answer and len(answer) > 10:
                partial_answers.append(answer)
        
        if not partial_answers:
            return "I couldn't find an answer to this question in the paper."
        
        # REDUCE: Combine answers
        combined = " ".join(partial_answers)
        
        # Return the combined answer
        return combined
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate summary using map-reduce"""
        chunks = self._chunk_text(text)
        
        if not chunks:
            return "Could not generate summary."
        
        partial_summaries = []
        for chunk in chunks[:3]:  # Limit to first 3 chunks
            prompt = "Summarize the following text concisely:"
            summary = self._generate_chunk_summary(chunk, prompt)
            if summary:
                partial_summaries.append(summary)
        
        if not partial_summaries:
            return "Could not generate a summary."
        
        return " ".join(partial_summaries)