## Real Estate RAG System

### Overview

A document Q&A system for real estate using Retrieval-Augmented Generation (RAG). Integrates real market data from Kaggle's USA Real Estate Dataset (2.2M properties) to answer questions about property listings, market trends, and investment metrics.

### Installation & Setup

```bash
# 1. Install Ollama
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull Mistral model
ollama pull mistral

# 3. Create environment
cd rag_realestate
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install dependencies
pip install langchain langchain-community langchain-ollama
pip install chromadb fastapi uvicorn pandas

# 5. Generate sample documents (or use Kaggle data)
python create_sample_docs.py

# OR download real data from Kaggle:
# https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
# python integrate_real_data.py
```

### Usage

**Command Line:**
```bash
# Index documents (first time only)
python rag_system.py

# Start API server
uvicorn api:app --reload --port 8000
```

**API Queries:**
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the median home price in Virginia?"}'

# Interactive docs
open http://localhost:8000/docs
```

**Example Response:**
```json
{
  "question": "What is the median home price in Virginia?",
  "answer": "The median home price in Virginia is $412,000 based on Q4 2024 data.",
  "sources": ["market_report_Virginia.txt"],
  "latency_ms": 2847.32
}
```

### Design Decisions & Learnings

**Chunk Size Trade-off:**
- Chose 500 characters per chunk with 50-character overlap
- Larger chunks preserve context but reduce retrieval precision
- Smaller chunks are more precise but can lose meaning
- This balance worked well for real estate Q&A

**Local vs API LLMs:**
- Used Ollama for zero cost and data privacy
- Latency: ~10-30 seconds on CPU
- Production would use GPU or API (vLLM for self-hosted, OpenAI for managed)

**Vector Database Choice:**
- ChromaDB: Simple, local, perfect for development
- Production would use Pinecone (managed) or Weaviate on K8s (self-hosted)

### Production Deployment Considerations

**Performance:**
- Replace Ollama with vLLM on GPU nodes (10-20x faster)
- Implement caching for common queries (Redis)
- Batch similar queries together

**Scalability:**
- Deploy on Kubernetes with horizontal pod autoscaling
- Separate API pods (CPU) from LLM serving pods (GPU)
- Use managed vector database (Pinecone/Weaviate Cloud)

**Reliability:**
- Add retry logic with exponential backoff
- Implement circuit breakers for failing services
- Graceful degradation when LLM is unavailable
- Comprehensive error handling

**Monitoring:**
- Track: query latency (P50, P95, P99), error rate, token usage
- Dashboards: Prometheus + Grafana
- Alerting: PagerDuty for critical failures
- Distributed tracing: OpenTelemetry

**Security:**
- API key authentication
- Rate limiting (per-user quotas)
- Input validation and sanitization
- HTTPS only

### File Structure

```
rag_realestate/
├── rag_system.py          # Core RAG logic
├── api.py                 # FastAPI endpoints
├── create_sample_docs.py  # Generate sample data
├── integrate_real_data.py # Kaggle data integration
├── documents/             # Source documents
├── chroma_db/            # Vector database storage
└── README.md
```
