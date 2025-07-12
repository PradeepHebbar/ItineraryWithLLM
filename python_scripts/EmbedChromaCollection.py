#!/usr/bin/env python3
"""
ChromaDB Collection Manager for Travel Embedding System

This module provides a simplified interface for managing vector embeddings
and document storage using ChromaDB and SentenceTransformers. It's designed
to work with travel-related data for the Mysuru itinerary planning system.

Key Features:
- SentenceTransformer embedding model setup
- In-memory ChromaDB client configuration
- Batch document addition with automatic embedding generation
- Progress tracking for embedding operations

Dependencies:
- sentence_transformers: For creating semantic embeddings
- chromadb: Vector database for similarity search

Configuration:
- Uses "all-MiniLM-L6-v2" model for embeddings (lightweight, efficient)
- Creates "mysore_places" collection for travel data
- In-memory storage (can be modified for persistence)

Author: Travel Itinerary System
Version: 1.0
"""

from sentence_transformers import SentenceTransformer
import chromadb

# === Embedding Model Configuration ===
# Using lightweight but effective model for travel data embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === ChromaDB Client Setup ===
# In-memory client for development/testing
# For production, add: persist_directory="db_dir" for data persistence
client     = chromadb.Client()
collection = client.create_collection(name="mysore_places")

def add_batch(texts, metadatas, ids):
    """
    Add a batch of documents to the ChromaDB collection with embeddings.
    
    This function takes a batch of text documents, generates embeddings for them,
    and adds them to the ChromaDB collection along with their metadata and IDs.
    
    Args:
        texts (list[str]): List of text documents to embed and index
        metadatas (list[dict]): List of metadata dictionaries corresponding to each text
        ids (list[str]): List of unique identifiers for each document
    
    Process:
        1. Generates embeddings for all texts using SentenceTransformer
        2. Converts embeddings to list format for ChromaDB compatibility
        3. Adds documents, embeddings, metadata, and IDs to the collection
    
    Raises:
        ValueError: If lengths of texts, metadatas, and ids don't match
        Exception: If embedding generation or database insertion fails
    
    Note:
        - Progress bar is shown during embedding generation
        - All input lists must have the same length
        - IDs must be unique within the collection
        - Metadata can contain arbitrary key-value pairs for filtering
    
    Example:
        >>> texts = ["Mysore Palace is a beautiful attraction", "Brindavan Gardens has musical fountains"]
        >>> metadatas = [{"type": "palace", "rating": 4.5}, {"type": "garden", "rating": 4.2}]
        >>> ids = ["palace_001", "garden_001"]
        >>> add_batch(texts, metadatas, ids)
    """
    embs = embedder.encode(texts, show_progress_bar=True)
    collection.add(
        embeddings=embs.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
