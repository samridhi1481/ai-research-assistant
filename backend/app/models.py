import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    authors = Column(String(500))
    filename = Column(String(200))
    page_count = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    abstract = Column(Text, nullable=True)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Database setup
engine = create_engine("sqlite:///data/papers.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)