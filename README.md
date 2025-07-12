# Travel Guide External Data - Project Backup

## Project Overview

This is a comprehensive backup of the Travel Guide External Data project - an AI-powered itinerary planning system for Mysuru (Mysore), India. The system uses RAG
(Retrieval-Augmented Generation) with multiple LLM backends (OpenAI GPT, Google Gemini, and Ollama) to generate personalized travel itineraries.

## What's Included in This Backup

### 📁 **python_scripts/**

Core application files:

- `itinerary_planner.py` - Main application (15KB) - The primary script for generating itineraries
- `CompleteEmbed_index.py` - Vector embedding and indexing for ChromaDB
- `EmbedChromaCollection.py` - ChromaDB collection management
- `FetchExternalData.py` - External data fetching utilities
- `PrepareExcelVector.py` - Excel data preprocessing for vector embeddings
- `PreparePDF.py` - PDF processing utilities
- `embed_index.py` - Legacy embedding indexing script
- `check_openai_models.py` - OpenAI model compatibility checker

### 📁 **prompts/**

LLM prompt templates for different backends:

- `Prompt.txt` - Base prompt template
- `Prompt_OpenAI.txt` - OpenAI-specific prompts
- `Prompt_Gemini.txt` - Google Gemini-specific prompts
- `Prompt_Ollama.txt` - Ollama-specific prompts
- `Prompt_Ollama_Simple.txt` - Simplified Ollama prompts

### 📁 **data/**

Essential data files:

- `mysuru_attractions.xlsx` - Comprehensive attractions database (116KB)
- `mysore_overview.pdf` - Travel guide PDF (1.6MB) _(if available)_

### 📁 **config/**

Configuration files:

- `.env.example` - Environment variables template
- `.env` - Current environment configuration _(contains sensitive data)_

### 📁 **old_versions/**

Previous versions and legacy code for reference

### 📁 **chromadb_db/**

Vector database files _(if available)_

## System Requirements

- **Python:** 3.9 or higher
- **Operating System:** macOS, Linux, or Windows
- **Memory:** Minimum 4GB RAM (8GB recommended)
- **Storage:** At least 2GB free space

## Key Dependencies

```
sentence-transformers
chromadb
openai
google-generativeai
python-dotenv
requests
pandas
openpyxl
PyPDF2
```

## Features

### 🎯 **Multi-LLM Support**

- **OpenAI GPT-4** - High-quality responses with token usage tracking
- **Google Gemini** - Cost-effective alternative with good performance
- **Ollama** - Local LLM execution for privacy

### 🔍 **RAG Implementation**

- Vector embeddings using SentenceTransformers
- ChromaDB for efficient similarity search
- Context-aware prompt generation

### 🎨 **Customizable Itineraries**

- Travel group types: friends, couples, solo, family
- Transport modes: car, public transport, bike
- Interest-based recommendations
- Configurable duration and starting point

### 📊 **Smart Data Processing**

- Excel spreadsheet integration
- PDF document processing
- Dynamic prompt assembly
- Token usage optimization

## How to Restore and Use

### 1. **Restore the Project**

```bash
chmod +x restore_script.sh
./restore_script.sh /path/to/new/location
```

### 2. **Set Up Environment**

```bash
cd /path/to/new/location
python -m venv venv_excel
source venv_excel/bin/activate  # On Windows: venv_excel\Scripts\activate
```

### 3. **Install Dependencies**

```bash
pip install sentence-transformers chromadb openai google-generativeai python-dotenv requests pandas openpyxl PyPDF2
```

### 4. **Configure API Keys**

Edit the `.env` file and add your API keys:

```
# Core LLM Services
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Data Collection Services (for FetchExternalData.py)
GOOGLE_API_KEY=your_google_places_api_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

**Get your API keys from:**

- **OpenAI:** https://platform.openai.com/api-keys
- **Google Gemini:** https://makersuite.google.com/app/apikey
- **Google Places API:** https://console.cloud.google.com/apis/credentials
- **RapidAPI:** https://rapidapi.com/ (for TripAdvisor integration)

### 5. **Initialize Database**

```bash
python CompleteEmbed_index.py
```

### 6. **Generate an Itinerary**

```bash
python itinerary_planner.py \
  --departure_from "Bangalore" \
  --destination "Mysuru Palace" \
  --days 3 \
  --transport_mode "car" \
  --interests "historical sites, local cuisine" \
  --travel_group "family" \
  --backend "openai"
```

## Usage Examples

### OpenAI Backend

```bash
python itinerary_planner.py \
  --departure_from "Bangalore" \
  --destination "Mysuru" \
  --days 2 \
  --transport_mode "public transport" \
  --interests "temples, gardens" \
  --travel_group "couples" \
  --backend "openai" \
  --model "gpt-4o"
```

### Gemini Backend

```bash
python itinerary_planner.py \
  --departure_from "Chennai" \
  --destination "Mysuru Palace" \
  --days 4 \
  --transport_mode "car" \
  --interests "architecture, shopping" \
  --travel_group "friends" \
  --backend "gemini" \
  --model "gemini-2.0-flash-exp"
```

### Ollama Backend (Local)

```bash
python itinerary_planner.py \
  --departure_from "Hyderabad" \
  --destination "Mysuru" \
  --days 1 \
  --transport_mode "bike" \
  --interests "nature, photography" \
  --travel_group "solo" \
  --backend "ollama" \
  --model "llama3:latest"
```

## Project Architecture

```
Travel Guide System
├── Data Layer
│   ├── Excel Database (Attractions)
│   ├── PDF Documents (Overview)
│   └── ChromaDB (Vector Store)
├── Processing Layer
│   ├── Vector Embeddings
│   ├── Semantic Search
│   └── Context Assembly
├── LLM Layer
│   ├── OpenAI GPT
│   ├── Google Gemini
│   └── Ollama (Local)
└── Output Layer
    ├── JSON Itineraries
    ├── Token Usage Stats
    └── Cost Estimates
```

## Troubleshooting

### Common Issues

1. **"Collection empty" error**

   - Run `python CompleteEmbed_index.py` to rebuild the vector database

2. **API key errors**

   - Check your `.env` file configuration
   - Verify API keys are valid and have sufficient credits

3. **Import errors**

   - Ensure all dependencies are installed
   - Activate the virtual environment

4. **ChromaDB issues**
   - Delete the `chromadb_db` folder and rebuild the index
   - Check file permissions

### Performance Tips

- Use `--K_places 5` for faster responses
- Choose `gemini` backend for cost-effective queries
- Use `ollama` for complete privacy (requires local setup)

## File Sizes and Performance

- **Total backup size:** ~2-3 MB (excluding ChromaDB)
- **ChromaDB size:** ~50-100 MB (varies by embeddings)
- **Average response time:** 3-10 seconds
- **Token usage:** 1,000-4,000 input tokens per query

## Security Notes

⚠️ **Important:** The `.env` file contains sensitive API keys. Keep this backup secure and don't share it publicly.

## Support and Updates

For the latest version and support:

- Check the original project directory
- Review the `Old/` folder for previous versions
- Test with different LLM backends for optimal results

---

**Created:** $(date +"%Y-%m-%d %H:%M:%S")  
**Backup Version:** v1.0  
**Project Status:** Working and tested
