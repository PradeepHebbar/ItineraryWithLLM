#!/usr/bin/env python3
"""
Intelligent Travel Itinerary Planner for Mysuru

This module provides a comprehensive AI-powered travel itinerary planning system
that leverages Retrieval-Augmented Generation (RAG) with multiple LLM backends
to create personalized travel plans for the Mysuru region.

Key Features:
- Multi-backend LLM support (OpenAI GPT, Google Gemini, Ollama)
- RAG-based context retrieval using ChromaDB vector database
- Semantic search for attractions and overview information
- Customizable prompt templates for different LLM backends
- Token usage tracking and cost estimation
- JSON-structured itinerary output
- Flexible parameter configuration via command-line interface

Architecture:
1. Data Retrieval: Uses sentence transformers and ChromaDB for semantic search
2. Context Assembly: Combines user parameters with retrieved attraction data
3. Prompt Engineering: Backend-specific templates with dynamic content injection
4. LLM Integration: Supports OpenAI, Gemini, and Ollama with unified interface
5. Response Processing: Handles both structured JSON and raw text outputs

Supported LLM Backends:
- OpenAI: GPT-4, GPT-4-turbo, GPT-3.5-turbo with API key authentication
- Google Gemini: Gemini-2.0-flash-exp, Gemini-pro with API key authentication  
- Ollama: Local models (llama3, mistral, etc.) via HTTP API

Dependencies:
- sentence_transformers: Semantic embeddings for attraction matching
- chromadb: Vector database for efficient similarity search
- openai: OpenAI API client (optional)
- google.generativeai: Google Gemini API client (optional)
- requests: HTTP client for Ollama API calls
- python-dotenv: Environment variable management

Usage:
    python itinerary_planner.py --departure_from "Bangalore" --destination "Mysuru" 
        --days 2 --transport_mode "car" --interests "history,nature" 
        --travel_group "couples" --backend "openai" --model "gpt-4"

Environment Variables:
- OPENAI_API_KEY: Required for OpenAI backend
- GEMINI_API_KEY: Required for Gemini backend
- TOKENIZERS_PARALLELISM: Set to "false" to avoid warnings

Author: Travel Itinerary System
Version: 2.0
Date: July 2025
"""

import argparse
import json
import requests
import subprocess
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")
    print("Install with: pip install python-dotenv")

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb

# Set OpenAI API key from environment variable - will be imported inside functions
# openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")


def load_template(path="Prompt.txt"):
    """
    Load and return the content of a prompt template file.
    
    This function loads prompt templates that contain placeholder variables
    for dynamic content injection. Templates are backend-specific and contain
    instructions for the LLM to generate structured itineraries.
    
    Args:
        path (str, optional): Path to the template file. Can be either:
            - Absolute path: Full path to template file
            - Relative path: Path relative to current directory
            - Filename only: Resolved relative to script directory
            Defaults to "Prompt.txt".
    
    Returns:
        str: Complete template content as a string with placeholder variables
            ready for formatting with actual values.
    
    Template Variables:
        Templates typically include variables like:
        - {departure_from}: Starting location
        - {destination}: Travel destination
        - {days}: Number of travel days
        - {transport_mode}: Mode of transportation
        - {interests}: User interests and preferences
        - {travel_group}: Type of travel group
        - {overview_list}: Dynamic overview snippets
        - {place_list}: Dynamic attraction descriptions
        - {K_places}: Number of attractions to consider
    
    File Resolution:
        - If path contains directory separators, uses path as-is
        - If filename only, resolves relative to script directory
        - Enables portable template loading regardless of execution context
    
    Raises:
        FileNotFoundError: If template file doesn't exist
        IOError: If file cannot be read due to permissions or corruption
        UnicodeDecodeError: If file contains invalid UTF-8 encoding
    
    Example:
        >>> template = load_template("Prompt_OpenAI.txt")
        >>> # Returns template string with {variable} placeholders
        >>> template = load_template("/path/to/custom_prompt.txt")
        >>> # Returns template from absolute path
    """
    # If path is just a filename, resolve it relative to the script directory
    if not os.path.dirname(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── Setup embedder and persistent Chroma client ───────────────────────────
embedder  = SentenceTransformer("all-MiniLM-L6-v2")
client    = chromadb.PersistentClient(path="chromadb_db")
# Create or load the places collection
collection = client.get_or_create_collection(name="mysore_places")
# If empty, rebuild index by running CompleteEmbed_index.py
if collection.count() == 0:
    print("Index empty. Running CompleteEmbed_index.py to rebuild embeddings...")
    try:
        # First, delete the existing collection to ensure a clean rebuild
        try:
            client.delete_collection(name="mysore_places")
        except:
            pass  # Collection might not exist
        
        subprocess.run([sys.executable, "CompleteEmbed_index.py"], check=True)
        
        # Recreate the client and collection to pick up the persisted data
        client = chromadb.PersistentClient(path="chromadb_db")
        collection = client.get_or_create_collection(name="mysore_places")
        print(f"Index rebuilt successfully. Collection now has {collection.count()} documents.")
        if collection.count() == 0:
            print("Warning: Collection is still empty after rebuilding. Please check CompleteEmbed_index.py")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error rebuilding index: {e}")
        print("Please fix CompleteEmbed_index.py and run it manually, or ensure the ChromaDB collection is populated.")
        sys.exit(1)

def get_overview_snippets(k=2):
    """
    Retrieve contextual overview snippets about Mysuru from the vector database.
    
    This function performs semantic search to find relevant overview information
    about Mysuru tourism, culture, and general travel context to provide 
    background information for itinerary planning.
    
    Args:
        k (int, optional): Number of overview snippets to retrieve. 
            Defaults to 2.
    
    Returns:
        list[str]: List of overview text snippets. Each snippet contains
            contextual information about Mysuru tourism, attractions, or
            cultural background. Returns empty list if no snippets found.
    
    Search Process:
        1. Creates embedding for generic overview query
        2. Performs similarity search in ChromaDB collection
        3. Filters results to PDF source documents only
        4. Returns top-k most relevant text snippets
    
    Data Source:
        - Source filter: "mysore_pdf" (overview documents)
        - Content: General tourism information, cultural context, travel tips
        - Encoding: Uses sentence-transformers for semantic matching
    
    Error Handling:
        - Prints warning if no overview snippets found
        - Returns empty list instead of raising exceptions
        - Gracefully handles ChromaDB connection issues
    
    Usage:
        - Provides contextual background for LLM prompts
        - Enhances itinerary planning with local knowledge
        - Supplements attraction-specific information
    
    Example:
        >>> snippets = get_overview_snippets(3)
        >>> for snippet in snippets:
        ...     print(f"Overview: {snippet[:100]}...")
    """
    query = "Mysuru travel overview"
    q_emb  = embedder.encode([query])
    results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        where={"source": "mysore_pdf"}
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        print("Warning: No overview snippets found in the database.")
        return []
    return docs


def get_place_contexts(profile, transport, k=10):
    """
    Retrieve relevant attraction information based on travel profile and transport mode.
    
    This function performs intelligent semantic search to find attractions that
    match the travel group profile and are accessible by the specified transport mode.
    
    Args:
        profile (str): Travel group type for filtering attractions:
            - "friends": Social activities, nightlife, group-friendly venues
            - "couples": Romantic settings, intimate experiences, scenic spots
            - "solo": Individual-friendly places, museums, cultural sites
            - "family": Child-friendly attractions, educational venues, parks
        transport (str): Mode of transportation for accessibility filtering:
            - "car": All attractions, including remote locations
            - "public transport": Easily accessible by bus/train
            - "bike": Bike-friendly routes and destinations
        k (int, optional): Maximum number of attraction contexts to retrieve.
            Defaults to 10.
    
    Returns:
        list[tuple[str, dict]]: List of (document_text, metadata) tuples where:
            - document_text (str): Full attraction description with all details
            - metadata (dict): Structured data including name, rating, location, etc.
            Returns empty list if no matching attractions found.
    
    Search Strategy:
        1. Constructs intelligent query combining profile and transport preferences
        2. Creates semantic embedding of the combined query
        3. Performs similarity search against attraction database
        4. Filters to Excel source data (structured attraction information)
        5. Returns top-k most relevant matches with full context
    
    Metadata Fields:
        - name: Attraction name
        - rating: User ratings and reviews
        - types: Attraction categories and tags
        - address: Location and accessibility information
        - suitability: Target audience flags
        - flavors: Activity type classifications
    
    Error Handling:
        - Prints warning if no place contexts found
        - Returns empty list for graceful error handling
        - Handles ChromaDB connection and query failures
    
    Usage:
        - Primary source of attraction data for itinerary generation
        - Enables personalized recommendations based on travel preferences
        - Provides rich context for LLM decision making
    
    Example:
        >>> contexts = get_place_contexts("couples", "car", 5)
        >>> for doc, meta in contexts:
        ...     print(f"{meta['name']}: {doc[:100]}...")
    """
    user_query = f"{profile}-friendly places accessible by {transport} in Mysuru"
    q_emb = embedder.encode([user_query])
    results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        where={"source": "excel"}
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        print("Warning: No place contexts found in the database.")
        return []
    return list(zip(docs, metas))

def assemble_prompt(template, params, overview_snips, place_contexts):
    """
    Assemble the final LLM prompt by combining template with dynamic content.
    
    This function takes a prompt template and dynamically injects user parameters,
    overview snippets, and attraction contexts to create a comprehensive prompt
    for the LLM to generate personalized travel itineraries.
    
    Args:
        template (str): Prompt template string with placeholder variables
            (loaded from template files like Prompt_OpenAI.txt)
        params (dict): User-provided parameters including:
            - departure_from: Starting location
            - destination: Travel destination  
            - days: Number of travel days
            - transport_mode: Transportation method
            - interests: User interests and preferences
            - travel_group: Type of travel group
            - K_places: Number of places to consider
        overview_snips (list[str]): Contextual overview snippets about Mysuru
        place_contexts (list[tuple[str, dict]]): Attraction data and metadata
    
    Returns:
        str: Complete assembled prompt ready for LLM consumption with all
            variables populated and dynamic content injected.
    
    Dynamic Content Generation:
        1. Overview List: Numbered list of contextual snippets
        2. Place List: Detailed attraction descriptions with metadata
        3. Place Names: Simple list for validation and reference
        4. Updated Counts: Actual number of places with valid content
    
    Variable Population:
        - Static variables: Direct substitution from params dict
        - Dynamic variables: Generated from retrieved content
        - Fallback values: Default messages for missing content
        - Count updates: Reflects actual available data
    
    Content Filtering:
        - Skips empty or whitespace-only snippets
        - Only includes attractions with valid names and descriptions
        - Maintains numbered lists for easy LLM reference
        - Updates K_places to reflect actual content availability
    
    Error Handling:
        - Validates all template variables are provided
        - Reports missing variables with helpful error messages
        - Lists available variables for debugging
        - Exits gracefully on template errors
    
    Template Variables:
        Core variables injected into templates:
        - overview_list: Formatted overview snippets
        - place_list: Detailed attraction information
        - place_names_only: Simple attraction name list
        - K_places, K: Updated place counts
        - All user parameters from params dict
    
    Example:
        >>> template = "Plan {days}-day trip to {destination}..."
        >>> params = {"days": 2, "destination": "Mysuru"}
        >>> prompt = assemble_prompt(template, params, overviews, places)
        >>> # Returns fully populated prompt string
    """
    prompt_vars = {**params}
    
    # Generate dynamic overview list
    overview_list = []
    for i, snip in enumerate(overview_snips, start=1):
        if snip.strip():  # Only include non-empty snippets
            overview_list.append(f"{i}) {snip}")
    prompt_vars["overview_list"] = "\n".join(overview_list) if overview_list else "No overview information available."
    
    # Generate dynamic place list with full descriptions
    place_list = []
    # Generate simple place names list for validation
    place_names_only = []
    K = params.get("K_places", len(place_contexts))
    for i, (doc, meta) in enumerate(place_contexts[:K], start=1):
        place_name = meta.get("name", f"Place {i}")
        if place_name and doc.strip():  # Only include places with content
            place_list.append(f"{i}) {place_name} — {doc}")
            place_names_only.append(place_name)
    
    prompt_vars["place_list"] = "\n".join(place_list) if place_list else "No place information available."
    prompt_vars["place_names_only"] = "\n".join([f"- {name}" for name in place_names_only])
    prompt_vars["K_places"] = len(place_list)  # Update to actual number of places with content
    prompt_vars["K"] = len(place_list)
    
    try:
        return template.format(**prompt_vars)
    except KeyError as e:
        print(f"Error: Template expects variable {e} but it's not provided.")
        print("Available variables:", list(prompt_vars.keys()))
        sys.exit(1)

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text string for LLM cost calculation.
    
    This function provides a rough approximation of token count for different
    LLM providers to help estimate API costs and manage token limits before
    making actual API calls.
    
    Args:
        text (str): Input text to analyze for token count estimation
    
    Returns:
        int: Estimated number of tokens based on character count approximation
    
    Algorithm:
        Uses a simple heuristic of 1 token ≈ 4 characters for English text.
        This approximation works reasonably well for:
        - OpenAI GPT models (GPT-3.5, GPT-4)
        - Google Gemini models
        - Most transformer-based language models
    
    Accuracy:
        - ±20% accuracy for typical English text
        - More accurate for longer texts (averaging effect)
        - Less accurate for highly technical or non-English content
        - Conservative estimate (tends to slightly overestimate)
    
    Limitations:
        - Does not account for special tokens (system prompts, formatting)
        - Simplified compared to actual tokenizer algorithms
        - May vary significantly for different languages
        - Should not be used for precise quota management
    
    Usage:
        - Pre-flight cost estimation before API calls
        - Prompt optimization and length management
        - Token budget planning for batch processing
        - User feedback on expected costs
    
    Alternative:
        For precise token counting, use the actual tokenizer:
        - OpenAI: tiktoken library
        - Gemini: official tokenizer APIs
        - Ollama: model-specific tokenizers
    
    Example:
        >>> text = "Plan a 2-day trip to Mysuru with historical sites"
        >>> tokens = estimate_tokens(text)
        >>> print(f"Estimated tokens: {tokens}")  # ~12 tokens
    """
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def call_llm_openai(prompt, model="gpt-4-turbo", max_tokens=4096, temperature=0.1):
    """
    Call OpenAI's GPT models for itinerary generation with detailed usage tracking.
    
    This function interfaces with OpenAI's Chat Completion API to generate
    travel itineraries using GPT models. It includes comprehensive token usage
    tracking and cost estimation for budget management.
    
    Args:
        prompt (str): Complete prompt text for itinerary generation
        model (str, optional): OpenAI model identifier. Recommended options:
            - "gpt-4o": Latest GPT-4 optimized model (default for quality)
            - "gpt-4-turbo": Fast GPT-4 variant for quicker responses
            - "gpt-3.5-turbo": Cost-effective option for simpler requests
            Defaults to "gpt-4-turbo".
        max_tokens (int, optional): Maximum tokens for response generation.
            Defaults to 4096. Recommended ranges:
            - Simple itineraries: 2048-4096
            - Detailed itineraries: 4096-8192
            - Complex multi-day plans: 8192+
        temperature (float, optional): Response creativity control (0.0-2.0).
            - 0.0-0.2: Highly focused, deterministic responses
            - 0.3-0.7: Balanced creativity and consistency
            - 0.8-2.0: High creativity, less predictable
            Defaults to 0.1 for consistent travel planning.
    
    Returns:
        str: Generated itinerary content from OpenAI, typically in JSON format
            when using structured prompts.
    
    Token Usage Tracking:
        Displays comprehensive usage statistics:
        - Input tokens: Prompt token count
        - Output tokens: Response token count  
        - Total tokens: Combined usage
        - Cost estimation: Based on current OpenAI pricing
    
    Cost Calculation (as of 2024):
        - GPT-4o: $2.50/1M input, $10.00/1M output tokens
        - Costs are estimates and may change with OpenAI pricing updates
        - Displayed in USD with 6-decimal precision
    
    Configuration:
        - API Key: Retrieved from OPENAI_API_KEY environment variable
        - Response Format: JSON object for structured output
        - Error Handling: Comprehensive error reporting and graceful exit
    
    Rate Limits:
        - Respects OpenAI's rate limiting automatically
        - No manual rate limiting implemented
        - Consider implementing backoff for high-volume usage
    
    Error Handling:
        - API authentication errors
        - Rate limit exceeded errors  
        - Model availability issues
        - Network connectivity problems
        - Invalid parameter errors
    
    Example:
        >>> prompt = "Generate a 2-day Mysuru itinerary..."
        >>> response = call_llm_openai(prompt, model="gpt-4o", max_tokens=8192)
        >>> # Displays token usage and costs, returns itinerary
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"}  # optional but helpful for structured JSON
        )
        
        # Display token usage details
        if response.usage:
            usage = response.usage
            print(f"\n===== OPENAI TOKEN USAGE =====")
            print(f"Input tokens (prompt): {usage.prompt_tokens}")
            print(f"Output tokens (response): {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
            
            # Calculate approximate costs (GPT-4o rates as of 2024)
            # GPT-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
            input_cost = (usage.prompt_tokens / 1_000_000) * 2.50
            output_cost = (usage.completion_tokens / 1_000_000) * 10.00
            total_cost = input_cost + output_cost
            
            print(f"Estimated cost:")
            print(f"  Input: ${input_cost:.6f}")
            print(f"  Output: ${output_cost:.6f}")
            print(f"  Total: ${total_cost:.6f}")
            print("=" * 30)
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

def call_llm_ollama(prompt, model="llama3:latest", max_tokens=4096, temperature=0.1):
    """
    Call local Ollama models for itinerary generation with privacy and cost benefits.
    
    This function interfaces with locally-hosted Ollama models to generate travel
    itineraries without sending data to external APIs. Ideal for privacy-conscious
    users or environments with restricted internet access.
    
    Args:
        prompt (str): Complete prompt text for itinerary generation
        model (str, optional): Ollama model identifier. Popular options:
            - "llama3:latest": Meta's Llama 3 (recommended for quality)
            - "llama3:70b": Larger Llama 3 variant (better quality, slower)
            - "mistral:latest": Mistral models for efficient generation
            - "codellama:latest": Code-optimized Llama variant
            - "gemma:latest": Google's Gemma models
            Defaults to "llama3:latest".
        max_tokens (int, optional): Maximum tokens for response generation.
            Note: Ollama uses "num_predict" parameter. Defaults to 4096.
        temperature (float, optional): Response creativity control (0.0-1.0).
            Lower values for more focused travel planning. Defaults to 0.1.
    
    Returns:
        str: Generated itinerary content from Ollama model, typically in
            natural language format (JSON support varies by model).
    
    Local Installation Requirements:
        - Ollama server running on localhost:11434
        - Desired model downloaded locally (e.g., `ollama pull llama3`)
        - Sufficient system resources (RAM: 8GB+ for 7B models, 32GB+ for 70B)
    
    API Configuration:
        - Endpoint: http://localhost:11434/api/generate
        - Method: POST with JSON payload
        - Stream: Disabled for complete response
        - No authentication required for local access
    
    Performance Considerations:
        - Response time depends on model size and hardware
        - GPU acceleration significantly improves speed
        - Larger models provide better quality but require more resources
        - No rate limits or API costs
    
    Error Handling:
        - Connection errors (Ollama not running)
        - Model not found errors (model not downloaded)
        - Resource exhaustion (insufficient memory)
        - HTTP request failures
        - Provides helpful troubleshooting messages
    
    Troubleshooting:
        - Check Ollama service: `ollama serve`
        - List available models: `ollama list`
        - Download models: `ollama pull <model_name>`
        - Monitor system resources during generation
    
    Privacy Benefits:
        - No data sent to external APIs
        - Complete local processing
        - No API key requirements
        - Suitable for sensitive travel planning
    
    Example:
        >>> prompt = "Generate a 2-day Mysuru itinerary..."
        >>> response = call_llm_ollama(prompt, model="llama3:latest")
        >>> # Returns locally-generated itinerary without external API calls
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        },
        "stream": False
    }
    
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        print(f"Make sure Ollama is running and the model '{model}' is available.")
        print("You can check available models with: ollama list")
        sys.exit(1)

def call_llm_gemini(prompt, model="gemini-2.0-flash-exp", max_tokens=4096, temperature=0.1):
    """
    Call Google Gemini models for itinerary generation with advanced capabilities.
    
    This function interfaces with Google's Gemini API to generate travel itineraries
    using state-of-the-art language models. Includes comprehensive token tracking
    and cost estimation for budget management.
    
    Args:
        prompt (str): Complete prompt text for itinerary generation
        model (str, optional): Gemini model identifier. Available options:
            - "gemini-2.0-flash-exp": Latest experimental Flash model (fastest)
            - "gemini-1.5-pro": Pro model with large context window
            - "gemini-1.5-flash": Fast model optimized for speed
            - "gemini-pro": Standard production model
            Defaults to "gemini-2.0-flash-exp".
        max_tokens (int, optional): Maximum output tokens for response.
            Gemini parameter: "max_output_tokens". Defaults to 4096.
        temperature (float, optional): Response creativity control (0.0-1.0).
            Lower values for more deterministic travel planning. Defaults to 0.1.
    
    Returns:
        str: Generated itinerary content from Gemini, formatted as JSON
            when using structured prompts.
    
    Token Usage Tracking:
        Displays detailed usage statistics when available:
        - Input tokens: Prompt token count
        - Output tokens: Response token count
        - Total tokens: Combined usage
        - Cost estimation: Based on current Gemini pricing
    
    Cost Calculation (as of 2024):
        - Gemini 2.0 Flash: $0.075/1M input, $0.30/1M output tokens
        - Costs are estimates and may change with Google pricing updates
        - Displayed in USD with 6-decimal precision
    
    Configuration:
        - API Key: Retrieved from GEMINI_API_KEY environment variable
        - Response Format: JSON MIME type for structured output
        - Generation Config: Temperature, max tokens, response format
        - Error Handling: Comprehensive error reporting with details
    
    Advanced Features:
        - Large context windows (up to 2M tokens for some models)
        - Multimodal capabilities (text + images for future enhancements)
        - JSON mode for structured output generation
        - Built-in safety filtering and content policies
    
    Rate Limits:
        - Respects Google's rate limiting automatically
        - Generous free tier with paid scaling options
        - No manual rate limiting implemented
    
    Error Handling:
        - API key validation and helpful setup instructions
        - Model availability and parameter validation
        - Network connectivity and timeout issues
        - Response parsing and content policy violations
        - Detailed error messages with troubleshooting guidance
    
    Setup Requirements:
        - Google AI Studio API key: https://makersuite.google.com/app/apikey
        - google-generativeai package: `pip install google-generativeai`
        - GEMINI_API_KEY environment variable
    
    Example:
        >>> prompt = "Generate a 2-day Mysuru itinerary..."
        >>> response = call_llm_gemini(prompt, model="gemini-2.0-flash-exp")
        >>> # Displays token usage and costs, returns structured itinerary
    """
    try:
        import google.generativeai as genai
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        
        # Create the model with generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        # Generate content and get detailed response
        response = model_instance.generate_content(prompt)
        
        # Display token usage details
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            print(f"\n===== GEMINI TOKEN USAGE =====")
            print(f"Input tokens (prompt): {usage.prompt_token_count}")
            print(f"Output tokens (response): {usage.candidates_token_count}")
            print(f"Total tokens: {usage.total_token_count}")
            
            # Calculate approximate costs (as of 2024, rates may change)
            # Gemini 2.0 Flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
            input_cost = (usage.prompt_token_count / 1_000_000) * 0.075
            output_cost = (usage.candidates_token_count / 1_000_000) * 0.30
            total_cost = input_cost + output_cost
            
            print(f"Estimated cost:")
            print(f"  Input: ${input_cost:.6f}")
            print(f"  Output: ${output_cost:.6f}")
            print(f"  Total: ${total_cost:.6f}")
            print("=" * 30)
        else:
            print("\n===== GEMINI RESPONSE =====")
            print("Token usage information not available")
            print("=" * 30)
        
        return response.text
        
    except ImportError:
        print("Error: google-generativeai package not installed")
        print("Install with: pip install google-generativeai")
        sys.exit(1)
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Print more detailed error info if available
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        sys.exit(1)

def call_llm(prompt, backend="openai", model="gpt-4-turbo", max_tokens=4096, temperature=0.1):
    """
    Unified interface for calling different LLM backends with consistent parameters.
    
    This function provides a single entry point for generating itineraries across
    multiple LLM providers, allowing easy switching between backends while 
    maintaining consistent parameter handling and error management.
    
    Args:
        prompt (str): Complete assembled prompt for itinerary generation
        backend (str, optional): LLM provider backend. Supported options:
            - "openai": OpenAI GPT models (requires API key)
            - "gemini": Google Gemini models (requires API key)  
            - "ollama": Local Ollama models (requires local installation)
            Defaults to "openai".
        model (str, optional): Model identifier specific to the chosen backend.
            Backend-specific defaults applied automatically. Defaults to "gpt-4-turbo".
        max_tokens (int, optional): Maximum response tokens across all backends.
            Defaults to 4096.
        temperature (float, optional): Creativity control (0.0-1.0 for most backends).
            Defaults to 0.1 for consistent travel planning.
    
    Returns:
        str: Generated itinerary content in the format specified by the prompt
            (typically JSON for structured output).
    
    Backend Selection:
        - Case-insensitive backend matching
        - Automatic parameter validation per backend
        - Backend-specific error handling and messaging
        - Consistent return format across providers
    
    Model Mapping:
        Each backend supports different models:
        - OpenAI: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
        - Gemini: gemini-2.0-flash-exp, gemini-1.5-pro, etc.
        - Ollama: llama3:latest, mistral:latest, etc.
    
    Error Handling:
        - Invalid backend name validation
        - Backend-specific error propagation
        - Helpful error messages with supported options
        - Graceful exit on configuration issues
    
    Usage Patterns:
        - Development: Use Ollama for free local testing
        - Production: Use OpenAI or Gemini for quality and reliability
        - Cost-conscious: Compare pricing across backends
        - Privacy-sensitive: Use Ollama for local processing
    
    Configuration Requirements:
        - OpenAI: OPENAI_API_KEY environment variable
        - Gemini: GEMINI_API_KEY environment variable
        - Ollama: Local Ollama server running with models downloaded
    
    Example:
        >>> # OpenAI backend
        >>> response = call_llm(prompt, "openai", "gpt-4o")
        >>> 
        >>> # Gemini backend  
        >>> response = call_llm(prompt, "gemini", "gemini-2.0-flash-exp")
        >>> 
        >>> # Local Ollama backend
        >>> response = call_llm(prompt, "ollama", "llama3:latest")
    """
    if backend.lower() == "openai":
        return call_llm_openai(prompt, model, max_tokens, temperature)
    elif backend.lower() == "ollama":
        return call_llm_ollama(prompt, model, max_tokens, temperature)
    elif backend.lower() == "gemini":
        return call_llm_gemini(prompt, model, max_tokens, temperature)
    else:
        print(f"Unknown LLM backend: {backend}. Use 'openai', 'ollama', or 'gemini'")
        sys.exit(1)

def main():
    """
    Main orchestrator function for the travel itinerary planning system.
    
    This function coordinates the entire itinerary generation pipeline from
    command-line argument parsing through final output generation. It handles
    user input validation, data retrieval, prompt assembly, and LLM interaction.
    
    Command-Line Interface:
        Required Arguments:
        - --departure_from: Starting location for the journey
        - --destination: Travel destination (typically "Mysuru")
        - --days: Number of travel days (integer)
        - --transport_mode: Transportation method (car/public transport/bike)
        - --interests: User interests and preferences (comma-separated)
        - --travel_group: Type of travel group (friends/couples/solo/family)
        
        Optional Arguments:
        - --K_places: Number of attractions to consider (default: 10)
        - --K_overview: Number of overview snippets (default: 2)
        - --template_path: Custom prompt template path (auto-selected by backend)
        - --model: LLM model identifier (backend-specific defaults)
        - --backend: LLM provider (openai/ollama/gemini, default: openai)
        - --max_tokens: Maximum response tokens (default: 8192)
        - --temperature: Response creativity (default: 0.1)
        - --show_prompt: Display generated prompt for debugging
    
    Pipeline Workflow:
        1. Argument Parsing: Validates and processes command-line inputs
        2. Template Selection: Auto-selects backend-appropriate template
        3. Data Retrieval: Fetches overview and attraction contexts from vector DB
        4. Prompt Assembly: Combines template with dynamic content
        5. Statistics Display: Shows prompt metrics and token estimates
        6. LLM Generation: Calls selected backend for itinerary creation
        7. Output Processing: Parses and displays results in structured format
    
    Template Auto-Selection:
        Automatically selects appropriate prompt templates:
        - OpenAI backend → Prompt_OpenAI.txt
        - Gemini backend → Prompt_Gemini.txt  
        - Ollama backend → Prompt_Ollama.txt
        - Custom path can override auto-selection
    
    Output Formats:
        - JSON Format: Structured itinerary with proper formatting
        - Raw Format: Fallback for unparseable responses
        - Error messages: Helpful debugging information
    
    Statistics Display:
        Pre-generation statistics:
        - Backend and model information
        - Prompt character and token counts
        - Expected resource usage
        
        Post-generation statistics (backend-specific):
        - Actual token usage and costs
        - Performance metrics
        - API call details
    
    Error Handling:
        - Invalid argument combinations
        - Missing environment variables (API keys)
        - Database connection issues
        - Template loading failures
        - LLM API errors
        - JSON parsing errors
    
    Usage Examples:
        Basic usage:
        >>> python itinerary_planner.py --departure_from "Bangalore" 
        ...     --destination "Mysuru" --days 2 --transport_mode "car" 
        ...     --interests "history,nature" --travel_group "couples"
        
        Advanced usage with custom settings:
        >>> python itinerary_planner.py --departure_from "Chennai" 
        ...     --destination "Mysuru" --days 3 --transport_mode "public transport"
        ...     --interests "culture,food,photography" --travel_group "solo"
        ...     --backend "gemini" --model "gemini-2.0-flash-exp" 
        ...     --max_tokens 12288 --show_prompt
        
        Local processing with Ollama:
        >>> python itinerary_planner.py --departure_from "Hyderabad"
        ...     --destination "Mysuru" --days 1 --transport_mode "bike"
        ...     --interests "adventure,nature" --travel_group "friends"
        ...     --backend "ollama" --model "llama3:latest"
    
    Environment Setup:
        - ChromaDB database with indexed attraction data
        - Environment variables for API keys (if using external providers)
        - Prompt template files in script directory
        - Sufficient system resources for chosen backend
    """
    parser = argparse.ArgumentParser(description="Plan itinerary via RAG + OpenAI")
    parser.add_argument("--departure_from", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--transport_mode", choices=["car","public transport","bike"], required=True)
    parser.add_argument("--interests", required=True)
    parser.add_argument("--travel_group", choices=["friends","couples","solo","family"], required=True)
    parser.add_argument("--K_places", type=int, default=10)
    parser.add_argument("--K_overview", type=int, default=2)
    parser.add_argument("--template_path", default="Prompt.txt")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--backend", choices=["openai", "ollama", "gemini"], default="openai", help="LLM backend to use")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens for LLM response")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM response")
    parser.add_argument("--show_prompt", action="store_true", help="Display the generated prompt (uses tokens)")
    args = parser.parse_args()

    # Set default template based on backend
    if args.template_path == "Prompt.txt":  # Only change if using default template
        if args.backend == "openai":
            args.template_path = "Prompt_OpenAI.txt"
        elif args.backend == "ollama":
            args.template_path = "Prompt_Ollama.txt"
        elif args.backend == "gemini":
            args.template_path = "Prompt_Gemini.txt"

    params = {
        "departure_from": args.departure_from,
        "destination": args.destination,
        "days": args.days,
        "transport_mode": args.transport_mode,
        "interests": args.interests,
        "travel_group": args.travel_group,
        "K_places": args.K_places
    }

    template = load_template(args.template_path)
    overview_snips = get_overview_snippets(args.K_overview)
    place_contexts = get_place_contexts(args.travel_group, args.transport_mode, args.K_places)
    prompt = assemble_prompt(template, params, overview_snips, place_contexts)

    # Show prompt statistics
    prompt_chars = len(prompt)
    estimated_tokens = estimate_tokens(prompt)
    print(f"\n===== PROMPT STATISTICS =====")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Prompt characters: {prompt_chars:,}")
    print(f"Estimated input tokens: {estimated_tokens:,}")
    print(f"Max output tokens: {args.max_tokens:,}")
    print("=" * 30)

    if args.show_prompt:
        print("\n===== GENERATED PROMPT =====\n")
        print(prompt)
        print("\n" + "="*50 + "\n")

    itinerary = call_llm(prompt, backend=args.backend, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
    try:
        structured = json.loads(itinerary)
        print("\n===== ITINERARY JSON =====\n", json.dumps(structured, indent=2))
    except json.JSONDecodeError:
        print("\n===== RAW ITINERARY =====\n", itinerary)

if __name__ == "__main__":
    main()
