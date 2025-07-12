#!/usr/bin/env python3
"""
Embedding and Indexing Module for Mysuru Travel Data

This module provides functionality to create vector embeddings and index data
from Excel spreadsheets and PDF documents into a ChromaDB collection for
semantic search and retrieval in the travel itinerary planning system.

Key Features:
- Loads attraction data from Excel files and overview data from PDF documents
- Creates vector embeddings using SentenceTransformers
- Indexes data in ChromaDB for efficient semantic search
- Processes data in batches for memory efficiency

Dependencies:
- pandas: Data manipulation and Excel file reading
- PyPDF2: PDF document processing
- sentence_transformers: Creating vector embeddings
- chromadb: Vector database for similarity search
- tqdm: Progress bar for batch processing

Author: Travel Itinerary System
Version: 1.0
"""

import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# === Data Loading and Preparation ===
df = pd.read_excel("mysuru_attractions.xlsx")
# TODO: build excel_texts from DataFrame rows (see PrepareExcelVector.py for reference)
excel_texts = []  # Placeholder - should contain text representations of Excel rows

reader = PdfReader("mysore_overview.pdf")
# TODO: build pdf_chunks from PDF pages with proper chunking strategy
pdf_chunks = []  # Placeholder - should contain [{"text": str, "id": str, "meta": dict}, ...]

# === Embedding Model and Database Setup ===
embedder  = SentenceTransformer("all-MiniLM-L6-v2")
client    = chromadb.Client()
collection = client.create_collection("mysore_places")

def index_excel():
    """
    Index Excel spreadsheet data into ChromaDB collection.
    
    This function processes attraction data from the Excel file, creates embeddings
    for each row, and adds them to the ChromaDB collection with metadata.
    
    Process:
    1. Creates unique IDs for each Excel row (format: "xls_{index}")
    2. Converts DataFrame to dictionary records for metadata
    3. Processes data in batches of 32 items for memory efficiency
    4. Creates embeddings using SentenceTransformer model
    5. Adds embeddings, documents, metadata, and IDs to ChromaDB collection
    
    Global Variables Used:
    - excel_texts: List of text representations of Excel rows
    - df: Pandas DataFrame containing attraction data
    - embedder: SentenceTransformer model for creating embeddings
    - collection: ChromaDB collection for storing indexed data
    
    Raises:
    - Exception: If Excel data is not properly loaded or embedder fails
    
    Note:
    - Batch size of 32 is optimized for memory usage and processing speed
    - Progress bar shows indexing progress for user feedback
    """
    ids      = [f"xls_{i}" for i in range(len(excel_texts))]
    metas    = df.to_dict(orient="records")
    for i in tqdm(range(0, len(excel_texts), 32), desc="Indexing Excel"):
        batch_texts = excel_texts[i:i+32]
        batch_ids   = ids[i:i+32]
        batch_metas = metas[i:i+32]
        embs = embedder.encode(batch_texts, show_progress_bar=False)
        collection.add(embeddings=embs.tolist(),
                       documents=batch_texts,
                       metadatas=batch_metas,
                       ids=batch_ids)

def index_pdf():
    """
    Index PDF document data into ChromaDB collection.
    
    This function processes text chunks from the PDF document, creates embeddings
    for each chunk, and adds them to the ChromaDB collection with metadata.
    
    Process:
    1. Extracts text, IDs, and metadata from PDF chunks
    2. Processes data in batches of 32 items for memory efficiency
    3. Creates embeddings using SentenceTransformer model
    4. Adds embeddings, documents, metadata, and IDs to ChromaDB collection
    
    Global Variables Used:
    - pdf_chunks: List of dictionaries containing PDF text chunks with structure:
                  [{"text": str, "id": str, "meta": dict}, ...]
    - embedder: SentenceTransformer model for creating embeddings
    - collection: ChromaDB collection for storing indexed data
    
    Raises:
    - Exception: If PDF data is not properly loaded or embedder fails
    - KeyError: If pdf_chunks don't have expected structure
    
    Note:
    - Batch size of 32 is optimized for memory usage and processing speed
    - Progress bar shows indexing progress for user feedback
    - PDF chunks should contain contextual information about Mysuru overview
    """
    texts = [c["text"] for c in pdf_chunks]
    ids   = [c["id"]   for c in pdf_chunks]
    metas = [c["meta"] for c in pdf_chunks]
    for i in tqdm(range(0, len(texts), 32), desc="Indexing PDF"):
        bt  = texts[i:i+32]
        bi  = ids[i:i+32]
        bm  = metas[i:i+32]
        embs = embedder.encode(bt, show_progress_bar=False)
        collection.add(embeddings=embs.tolist(),
                       documents=bt,
                       metadatas=bm,
                       ids=bi)

if __name__ == "__main__":
    """
    Main execution block for the embedding and indexing process.
    
    This section orchestrates the complete indexing workflow:
    1. Indexes Excel data containing Mysuru attraction information
    2. Indexes PDF data containing overview and contextual information
    3. Provides completion confirmation
    
    Prerequisites:
    - mysuru_attractions.xlsx file must exist in the current directory
    - mysore_overview.pdf file must exist in the current directory
    - excel_texts and pdf_chunks must be properly populated before execution
    
    Output:
    - ChromaDB collection "mysore_places" populated with indexed data
    - Success message indicating completion
    
    Note:
    - This script should be run after data preparation steps are completed
    - The indexed data will be available for semantic search in the itinerary planner
    """
    index_excel()
    index_pdf()
    print("✅ Indexing complete")
