#!/usr/bin/env python3
"""
Excel Data Vectorization Preparation Module

This module processes Excel spreadsheet data containing Mysuru attraction information
and converts it into text representations suitable for vector embedding. It's designed
to work with the travel itinerary planning system's data pipeline.

Key Features:
- Loads attraction data from Excel files
- Converts structured data rows into coherent text descriptions
- Optimizes text format for semantic embedding models
- Handles missing data gracefully

Data Schema Expected:
- name: Attraction name
- description: Detailed description
- types: Comma-separated attraction types
- address: Physical address
- lat/lng: Geographic coordinates
- rating: User rating
- review_count: Number of reviews
- latest_reviews: Recent review snippets
- sentiment_score: Sentiment analysis score
- distance_km: Distance from city center
- opening_hours: Operating hours
- Tags: Feature tags
- suitability: Target audience flags

Dependencies:
- pandas: Excel file reading and data manipulation

Author: Travel Itinerary System
Version: 1.0
"""

import pandas as pd

# === Data Loading ===
# Load the main attractions dataset
df = pd.read_excel("mysuru_attractions.xlsx")

def row_to_text(r):
    """
    Convert a DataFrame row to a coherent text representation for embedding.
    
    This function takes a pandas Series (DataFrame row) containing attraction data
    and converts it into a natural language text that captures all relevant
    information for semantic search and retrieval.
    
    Args:
        r (pandas.Series): A row from the attractions DataFrame containing attraction data
    
    Returns:
        str: A formatted text string containing all attraction information
    
    Text Format:
        The output follows a structured format with labeled fields:
        "Name: {name}. Description: {description}. Types: {types}. ..."
        
    Handles Missing Data:
        - Uses .get() method with empty string defaults
        - Ensures no None values break the text formatting
        - Maintains consistent structure even with incomplete data
    
    Fields Included:
        - Basic info: name, description, types, address
        - Location: latitude, longitude, distance from center
        - Quality metrics: rating, review count, sentiment score
        - Operational: opening hours
        - Classification: tags, suitability flags
        - User feedback: latest reviews
    
    Example:
        >>> row = df.iloc[0]
        >>> text = row_to_text(row)
        >>> print(text)
        Name: Mysore Palace. Description: A magnificent royal palace...
    """ 

# 1. Load your spreadsheet
df = pd.read_excel("mysuru_attractions.xlsx")

# 2. For each row, build a single “document text” that the embedding model will consume.
def row_to_text(r):
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
        f"Visit duration: {r.get('opening_hours','')}. "
        f"Flavors: {r.get('Tags','')}. "
        f"Suitable for: {r.get('suitability','')}. "
    )

# === Text Conversion Processing ===
# Convert all DataFrame rows to text representations for embedding
excel_texts = [row_to_text(row) for _, row in df.iterrows()]

# === Module Usage Notes ===
"""
Usage Example:
    1. Ensure mysuru_attractions.xlsx exists in the current directory
    2. Import this module: from PrepareExcelVector import excel_texts
    3. Use excel_texts list for embedding in other modules
    
Integration:
    - This module is typically used by embed_index.py for vectorization
    - The excel_texts list should be passed to embedding functions
    - Text format is optimized for SentenceTransformer models
    
Output:
    - excel_texts: List of strings, one per attraction
    - Each string contains all relevant attraction information
    - Format is consistent and embedding-model friendly
"""
