#!/usr/bin/env python3
"""
Complete Embedding Index Builder for Mysuru Travel Data

This script builds a complete vector database index from multiple data sources
(Excel spreadsheets and PDF documents) for the Mysuru travel itinerary planner.
It processes and embeds all travel-related data into ChromaDB for efficient
semantic search and retrieval-augmented generation (RAG).

Features:
- Excel data processing with comprehensive place information
- PDF document chunking with overlap for better context preservation
- Batch processing for memory efficiency
- Persistent ChromaDB storage with automatic cleanup
- Progress tracking with tqdm

Data Sources:
- mysuru_attractions.xlsx: Comprehensive place database with ratings, reviews, etc.
- mysore_overview.pdf: General travel information and city overview

Author: Travel Planning System
Date: 2025
Version: 1.0
"""

import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb
from tqdm import tqdm

# ——— Step 1: Prepare Excel texts ——————————————————————————————
"""
Load and process Excel data containing Mysuru attractions information.
This section reads the comprehensive attractions database and converts
each row into a formatted text suitable for embedding.
"""
df = pd.read_excel("mysuru_attractions.xlsx")

def row_to_text(r):
    """
    Convert a pandas DataFrame row to a formatted text string for embedding.
    
    This function takes a row from the attractions DataFrame and creates
    a comprehensive text description that includes all relevant information
    about a place. The format is optimized for semantic search and LLM
    understanding.
    
    Args:
        r (pandas.Series): A row from the attractions DataFrame containing
                          place information like name, description, types, etc.
    
    Returns:
        str: Formatted text string containing all place information
        
    Example:
        >>> row = df.iloc[0]
        >>> text = row_to_text(row)
        >>> print(text[:100])
        Name: Mysuru Palace. Description: Historical palace with Indo-Saracenic...
    """
    return (
        f"Name: {r['name']}. "
        f"Description: {r.get('description','')}. "
        f"Types: {r.get('types','')}. "
        f"Address: {r.get('address','')}. "
        f"Latitude: {r.get('lat','')}. "
        f"Longitude: {r.get('lng','')}. "
        f"Rating: {r.get('rating','')}. "
        f"Total_Reviews: {r.get('review_count','')}. "
        f"Latest_Reviews: {r.get('latest_reviews','')}. "
        f"Sentiment_score: {r.get('sentiment_score','')}. "
        f"Distance_From_Center: {r.get('distance_km','')}. "
        f"Visit_duration: {r.get('approximate_visit_duration','')}. "
        f"Hours: {r.get('opening_hours','')}. "
        f"Flavors: {r.get('flavors','')}. "
        f"Suitable_for: {r.get('suitability','')}."
    )

# Process Excel data into embeddings format
excel_texts = [row_to_text(row) for _, row in df.iterrows()]
excel_metadatas = df.to_dict(orient="records")
for m in excel_metadatas:
    m['source'] = 'excel'

# ——— Step 2: Chunk the PDF ——————————————————————————————————
"""
Process PDF documents containing Mysuru travel overview information.
This section reads PDF files, extracts text, and chunks them into
manageable pieces for embedding while preserving context through overlap.
"""
reader = PdfReader("mysore_overview.pdf")
pages = [p.extract_text() for p in reader.pages]

def chunk_text(text, size=500, overlap=100):
    """
    Split text into overlapping chunks for better context preservation.
    
    This function divides long text into smaller chunks with overlap between
    consecutive chunks. This approach helps maintain context across chunk
    boundaries, which is important for semantic search accuracy.
    
    Args:
        text (str): Input text to be chunked
        size (int): Maximum characters per chunk (default: 500)
        overlap (int): Number of characters to overlap between chunks (default: 100)
    
    Returns:
        list[str]: List of text chunks with specified overlap
        
    Example:
        >>> text = "A very long document about Mysuru travel..."
        >>> chunks = chunk_text(text, size=100, overlap=20)
        >>> print(f"Created {len(chunks)} chunks")
        Created 5 chunks
    """
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

# Create structured PDF chunks with metadata
pdf_chunks = []
for pi, pg in enumerate(pages):
    for ci, chunk in enumerate(chunk_text(pg)):
        pdf_chunks.append({
            "id": f"pdf_{pi}_{ci}",
            "text": chunk,
            "meta": {"source": "mysore_pdf", "page": pi, "chunk_id": ci}
        })

# ——— Step 3: Embedder & persistent Chroma client ————————————————————
"""
Initialize the embedding model and ChromaDB client.
This section sets up the SentenceTransformer model for creating vector
embeddings and configures ChromaDB for persistent vector storage.
"""
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chromadb_db")

# Delete existing collection if it exists to ensure clean rebuild
try:
    client.delete_collection(name="mysore_places")
    print("Deleted existing collection.")
except:
    pass  # Collection might not exist

collection = client.get_or_create_collection(name="mysore_places")

# ——— Steps 4–5: Index the Excel rows —————————————————————————
def index_excel():
    """
    Index Excel attraction data into ChromaDB with batch processing.
    
    This function processes the Excel data in batches to efficiently
    create embeddings and store them in the ChromaDB collection.
    It handles memory management by processing data in smaller chunks.
    
    Batch Processing:
    - Processes 32 rows at a time for optimal memory usage
    - Creates embeddings for each batch
    - Stores embeddings with associated metadata in ChromaDB
    
    Progress Tracking:
    - Uses tqdm to show indexing progress
    - Displays batch processing status
    
    Example:
        >>> index_excel()
        Indexing Excel: 100%|██████████| 3/3 [00:05<00:00,  1.67it/s]
    """
    batch_size = 32
    for i in tqdm(range(0, len(excel_texts), batch_size), desc="Indexing Excel"):
        bt = excel_texts[i:i+batch_size]
        bi = [f"xls_{j}" for j in range(i, min(i+batch_size, len(excel_texts)))]
        bm = excel_metadatas[i:i+batch_size]
        embs = embedder.encode(bt, show_progress_bar=False)
        collection.add(
            embeddings=embs.tolist(),
            documents=bt,
            metadatas=bm,
            ids=bi
        )

# ——— Step 6: Index the PDF chunks ——————————————————————————
def index_pdf():
    texts = [c["text"] for c in pdf_chunks]
    ids   = [c["id"]   for c in pdf_chunks]
    metas = [c["meta"] for c in pdf_chunks]
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Indexing PDF"):
        bt  = texts[i:i+batch_size]
        bi  = ids[i:i+batch_size]
        bm  = metas[i:i+batch_size]
        embs = embedder.encode(bt, show_progress_bar=False)
        collection.add(
            embeddings=embs.tolist(),
            documents=bt,
            metadatas=bm,
            ids=bi
        )

# ——— Main: run indexing and persist ——————————————————————————
if __name__ == "__main__":
    index_excel()
    index_pdf()
    # client.persist()  # ensure vectors are saved to disk
    print("✅ Done, total vectors:", collection.count())
