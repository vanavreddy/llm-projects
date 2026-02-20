# LLM Projects Portfolio

Two hands-on projects demonstrating practical LLM application development.

---

## Project 1: Real Estate RAG System

### Overview

A document Q&A system for real estate using Retrieval-Augmented Generation (RAG). Integrates real market data from Kaggle's USA Real Estate Dataset (2.2M properties) to answer questions about property listings, market trends, and investment metrics.

### Architecture

```
User Question
    ↓
FastAPI Endpoint (/query)
    ↓
Embedding Generation (Ollama/Mistral)
    ↓
ChromaDB Vector Search (top 3 chunks via cosine similarity)
    ↓
Retrieved Context + Question → LLM (Mistral 7B)
    ↓
Answer + Sources + Latency
```

### Technology Stack

- **LLM**: Mistral 7B (via Ollama, local inference)
- **Embeddings**: Ollama embeddings (local)
- **Vector Database**: ChromaDB (local storage)
- **Orchestration**: LangChain
- **API Framework**: FastAPI
- **Data Source**: Kaggle USA Real Estate Dataset

### Key Features

- **Real Data Integration**: Uses actual property listings from 2.2M property dataset
- **Semantic Search**: Finds relevant information even when exact keywords don't match
- **Source Attribution**: Returns which documents were used to generate the answer
- **Latency Tracking**: Monitors query performance
- **Configurable Chunking**: 500-character chunks with 50-character overlap


## Project 2: LinkedIn Email Agent

### Overview

An intelligent agent that processes LinkedIn job alert emails, extracts job listings, and can be extended to rate jobs against your profile. Solves the real problem of manually scanning dozens of daily job emails.

### Architecture

```
Gmail API (OAuth 2.0)
    ↓
Fetch Emails (from:jobalerts-noreply@linkedin.com)
    ↓
Extract Plain Text Body
    ↓
Classification (Job Alert vs Post vs Other)
    ↓
Pattern Matching Extraction:
  - Find "View job:" lines as anchors
  - Work backwards to extract Title, Company, Location
  - Extract salary if present
    ↓
Structured Job List Output
```

### Technology Stack

- **Email Access**: Gmail API (OAuth 2.0)
- **LLM**: Mistral 7B (via Ollama) for classification and future matching
- **Parsing**: Regex + pattern matching
- **Language**: Python 3.11

### Key Features

- **OAuth Authentication**: Secure Gmail access with token persistence
- **Email Classification**: Distinguishes job alerts from posts and other notifications
- **Robust Extraction**: Handles LinkedIn's email format variations
- **Salary Detection**: Extracts compensation when available
- **Clean Output**: Formatted job listings with all key information


## Repository Structure

```
llm_projects/
├── README.md                    # This file
├── environment.yml              # Conda environment
├── rag_realestate/
│   ├── rag_system.py
│   ├── api.py
│   ├── create_sample_docs.py
│   ├── integrate_real_data.py
│   ├── documents/
│   └── chroma_db/
└── agent_summarizer/
    ├── gmail_agent_fixed.py
    ├── gmail_agent_with_matching.py
    ├── credentials.json (not in git)
    └── token.json (not in git)
```

---


## License

Personal portfolio projects - feel free to reference or learn from, but not for commercial use.

---

## Acknowledgments

- Kaggle for the USA Real Estate Dataset
- Ollama for making local LLM inference accessible
- LangChain for RAG abstractions
- Google for Gmail API
