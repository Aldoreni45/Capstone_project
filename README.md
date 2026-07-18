# Research Paper Answer Bot - CRAG Edition

A production-grade **Corrective Retrieval-Augmented Generation (CRAG)** chatbot with advanced query routing, automatic retrieval correction, web search fallback, and comprehensive evaluation framework.

**Dual UI Support**: Streamlit dashboard for development/testing and modern Next.js frontend for production use.

---

## 🚀 Key Features

- **CRAG Pipeline**: Automatic retrieval quality evaluation (GOOD/PARTIAL/BAD)
- **Query Rewriting**: LLM-powered query optimization for improved retrieval
- **Web Search Fallback**: DuckDuckGo integration for external context
- **Context Merging**: Intelligent combination of document and web sources
- **Query Routing**: Smart classification (greetings, general knowledge, document queries)
- **9 Retrieval Strategies**: Cosine, MMR, Hybrid, Multi-Query, Parent, Compression, Ensemble, Self-Query, Threshold
- **Cross-Encoder Reranking**: Advanced relevance scoring with MiniLM-L-6-v2
- **Conversational Memory**: Buffer, Token Window, and Summary memory types
- **Automatic Citations**: Metadata-based citation generation from retrieved chunks
- **LangSmith Tracing**: Comprehensive pipeline monitoring and debugging
- **Production-Grade**: Modular architecture with comprehensive logging and error handling
- **FastAPI Backend**: RESTful API for production deployments
- **Next.js Frontend**: Modern React-based UI with TypeScript and Tailwind CSS

---

## 🏗️ Architecture Overview

```
                    ┌────────────────────────┐
                    │      Streamlit UI      │
                    │  (Dev/Test Dashboard)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
        ┌───────────────────┐    ┌───────────────────┐
        │   Next.js UI      │    │   FastAPI Backend  │
        │  (Production)     │    │   (REST API)       │
        └───────────┬─────────┘    └───────────┬─────────┘
                    │                          │
                    └───────────┬──────────────┘
                                ▼
                    ┌────────────────────────┐
                    │   Query Classifier     │
                    │ (Greeting/General/Doc) │
                    └───────────┬────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
        ┌───────────────────┐    ┌───────────────────┐
        │  General Chat/Knowledge│    │  CRAG RAG Pipeline  │
        └───────────────────┘    └───────────┬─────────┘
                                            │
                                ┌───────────┴────────────┐
                                ▼                        ▼
                    ┌───────────────────┐    ┌───────────────────┐
                    │  Initial Retrieval │    │  Retrieval Evaluator│
                    │  (Weaviate + 9     │    │  (GOOD/PARTIAL/BAD)│
                    │   Strategies)       │    └───────────┬─────────┘
                    └───────────┬─────────┘                │
                                │                          │
                    ┌───────────┴────────────┬──────────────┘
                    ▼                        ▼
        ┌────────────────———─┐    ┌───────────────────┐
        │  Query Rewriter    │    │  Web Searcher     │
        │  (Groq LLM)         │    │  (DuckDuckGo)     │
        └───────────┬─────────┘    └───────────┬─────────┘
                    │                          │
                    └───────────┬──────────────┘
                                ▼
                    ┌───────────────────┐
                    │  Context Merger    │
                    │  (Docs + Web)      │
                    └───────────┬─────────┘
                                ▼
                    ┌───────────────────┐
                    │  Reranker         │
                    │  (Cross-Encoder)  │
                    └───────────┬─────────┘
                                ▼
                    ┌───────────────────┐
                    │  Context Compressor│
                    └───────────┬─────────┘
                                ▼
                    ┌───────────────────┐
                    │  Groq LLM          │
                    │  (Answer Generation)│
                    └───────────┬─────────┘
                                ▼
                    ┌───────────────────┐
                    │  Citation Generator│
                    │  (Metadata-based)  │
                    └───────────────────┘
```

---

## 📁 Folder Structure

```
capstone_project/
├── api/                    # FastAPI REST API backend
│   ├── __init__.py
│   └── app.py             # Main FastAPI application with endpoints
├── frontend/               # Next.js frontend application
│   ├── src/
│   │   ├── app/           # Next.js app router pages
│   │   ├── components/    # React components (ChatTab, DocumentsTab, etc.)
│   │   ├── lib/           # API client and utilities
│   │   └── types/         # TypeScript type definitions
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
├── .streamlit/             # Streamlit theme configurations
├── config/                 # YAML config and settings loaders
├── custom_logging/         # Structured logs with rotation and retries
├── pydantic_models/        # Strictly typed input/output schemas
├── loaders/                # PDF parsing and metadata extractors
├── chunking/               # Recursive and Semantic splitting strategies
├── embeddings/             # HuggingFace BGE (large & small) embedding wrappers
├── vectordb/               # Weaviate indexing and client CRUD wrappers
├── retrievers/             # 9 retrieval engines (Cosine, MMR, Hybrid, MultiQuery,
│                           #   Parent, Compression, Ensemble, SelfQuery, Threshold)
├── rerankers/              # Cross-Encoder (MiniLM-L-6-v2) reranker
├── memory/                 # Session-based, Buffer, Summary and Token memories
├── prompts/                # Optimized templates for comprehensive responses
├── query/                  # Query classification and routing modules
│   ├── classifier.py       # Conservative query classifier
│   ├── general_chat.py    # Greeting and small talk handler
│   └── general_knowledge.py # General knowledge handler
├── crag/                   # CRAG (Corrective RAG) components
│   ├── retrieval_evaluator.py # GOOD/PARTIAL/BAD classification
│   ├── query_rewriter.py   # LLM-powered query rewriting
│   ├── web_search.py       # DuckDuckGo web search fallback
│   └── context_merger.py   # Document + web context merging
├── chains/                 # Master orchestrator CRAG-enhanced RAG Pipeline
├── evaluation/             # Unified RAGAS and DeepEval metrics runners
├── streamlit/              # Streamlit dashboard with CRAG status indicators
├── utils/                  # LangSmith tracing, citation generator, metrics
├── tests/                  # Pytest test cases
└── docker/                 # Production Docker and Docker Compose files
```

---

## 🔧 Technical Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.12+ · TypeScript |
| **Orchestration** | LangChain >= 0.3 · PydanticAI |
| **CRAG Components** | Custom Retrieval Evaluator · Query Rewriter · Web Searcher · Context Merger |
| **Embeddings** | HuggingFace BGE — `BAAI/bge-large-en-v1.5` (default, 1024-dim) · `BAAI/bge-small-en-v1.5` (384-dim) — both run **100% locally**, no API key required |
| **Vector Database** | Weaviate Cloud (hosted) with cosine & hybrid search |
| **LLM Inference** | Groq Cloud — Llama-3.3-70B · DeepSeek-R1-70B · Gemma-2-9B · Qwen-2.5-Coder · Mixtral-8x7B |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **Web Search** | DuckDuckGo API (no API key required) |
| **Evaluation** | RAGAS · DeepEval |
| **Backend API** | FastAPI · Uvicorn |
| **Frontend UI** | Next.js 14 · React 18 · TypeScript · Tailwind CSS · Lucide Icons |
| **Dev UI** | Streamlit · Plotly |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) (Python) · npm (Node.js) |

---

## 🎯 CRAG Workflow

The CRAG pipeline automatically corrects retrieval quality through intelligent evaluation and fallback mechanisms:

### Retrieval Quality Classification

- **GOOD**: High relevance scores (>0.7), sufficient chunks (≥3), diverse sources → Proceed with normal RAG
- **PARTIAL**: Moderate scores (0.5-0.7), limited chunks (1-2) → Rewrite query and retry retrieval
- **BAD**: Low scores (<0.5), no chunks, or poor relevance → Web search fallback + document supplementation

### Correction Strategies

1. **Query Rewriting**: Groq LLM optimizes queries with domain terms, expanded abbreviations, and contextual information
2. **Web Search Fallback**: DuckDuckGo provides external context when document retrieval fails
3. **Context Merging**: Intelligent combination of document and web sources with multiple strategies
4. **Automatic Citations**: Metadata-based citation generation from retrieved chunks only

### CRAG Status Indicators

The Streamlit UI displays real-time CRAG status:
- 🔄 **Query Rewritten**: When query optimization was triggered
- 🌐 **Web Search Enhanced**: When external context was used
- **Retrieval Quality**: Color-coded badge (Green=GOOD, Orange=PARTIAL, Red=BAD)

---

## 🔍 Retrieval Strategies

Nine retrieval engines are available and switchable at runtime from the Streamlit sidebar:

| Strategy | Key | Description |
|---|---|---|
| Cosine Similarity | `cosine` | Standard dense vector similarity search |
| MMR | `mmr` | Maximal Marginal Relevance — balances relevance & diversity |
| Hybrid | `hybrid` | Fuses dense embeddings + BM25 sparse scores |
| Multi-Query | `multiquery` | LLM generates query variants; results are union-merged |
| Parent Document | `parent` | Retrieves small chunks, expands to full parent context |
| Contextual Compression | `compression` | LLM compresses retrieved docs to answer-relevant snippets |
| Ensemble | `ensemble` | Weighted fusion of Cosine + MMR + Hybrid |
| Self-Query | `selfquery` | LLM parses query into structured metadata filters |
| Similarity Threshold | `threshold` | Returns only results above a confidence cutoff |

---

## 💻 Local Setup

### 1. Prerequisites

- Python **3.12+** (managed automatically by `uv`)
- [uv](https://github.com/astral-sh/uv) package manager

Install `uv` if you don't have it:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment & Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd capstone_project

# Create a local .venv with Python 3.12
uv venv .venv --python 3.12

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

> **Note:** The first run will download HuggingFace models (`BAAI/bge-large-en-v1.5` ~1.3GB and the cross-encoder ~90MB) automatically from the HuggingFace Hub. Subsequent runs use the local cache.

### 3. Environment Configuration

Copy the environment template and fill in your keys:
```bash
cp .env.example .env
```

Open `.env` and set the following variables:

| Variable | Required | Description |
|---|---|---|
| `WEAVIATE_URL` | Yes | Your Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | Yes | Weaviate Cloud API key |
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `HF_TOKEN` | Optional | HuggingFace token for model downloads |
| `LANGCHAIN_API_KEY` | Optional | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Set to `true` to enable LangSmith traces |

**Frontend Environment Variables** (for Next.js):
| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | No | FastAPI backend URL (default: `http://localhost:8000`) |

> **No OpenAI API key is required.** All embeddings run locally via HuggingFace.

### 4. Running the Application

#### Option 1: Streamlit Dashboard (Development/Testing)

Launch the Streamlit dashboard:
```bash
streamlit run streamlit/app.py
```

The app will be available at **http://localhost:8501**.

#### Option 2: FastAPI Backend + Next.js Frontend (Production)

Start the FastAPI backend:
```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000** with auto-generated docs at **http://localhost:8000/docs**.

Start the Next.js frontend:
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at **http://localhost:3000**.

### 5. Running Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

---

## 🌐 API Endpoints

The FastAPI backend provides the following REST endpoints:

### Health Check
```http
GET /health
```
Returns the health status of the RAG pipeline.

### Query
```http
POST /query
Content-Type: application/json

{
  "query": "What is attention mechanism?",
  "session_id": "optional-session-id",
  "mode": "rag",
  "memory_type": "buffer"
}
```
Returns the answer with citations, confidence score, and retrieval metadata.

### Document Ingestion
```http
POST /ingest
Content-Type: multipart/form-data

file: <PDF file>
```
Ingests a PDF document into the vector database.

### API Documentation
Interactive API documentation available at **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/redoc** (ReDoc).

---

## 🎨 Frontend Features

The Next.js frontend provides a modern, responsive UI with:

- **Chat Tab**: Real-time chat interface with message history and context display
- **Documents Tab**: PDF upload and ingestion interface
- **Metrics Tab**: Performance metrics and latency visualization
- **Evaluation Tab**: RAG evaluation metrics dashboard
- **Settings Sidebar**: Configuration options for embedding models, retrieval strategies, and LLM selection
- **Dark Theme**: Modern dark mode UI with gradient accents
- **Responsive Design**: Works on desktop and mobile devices

---

## ⚙️ Configuration

All model settings are controlled from `config/config.yaml`. Key sections:

```yaml
embeddings:
  default_model: "huggingface"   # Options: "huggingface", "bge"
  huggingface:
    model_name: "BAAI/bge-large-en-v1.5"
    dimension: 1024
    device: "cpu"                # Change to "cuda" for GPU acceleration

llm:
  default_provider: "groq"
  groq:
    default_model: "llama-3.3-70b-versatile"
    temperature: 0.0
    max_tokens: 2048              # Increased for comprehensive responses

retrieval:
  bm25:
    k1: 1.5                       # BM25 parameter for hybrid retrieval
    b: 0.75
```

No code changes are needed to switch embedding models or LLM providers — edit the YAML and restart.

---

## 🐳 Docker Deployment

Build and deploy the containerized application using Docker Compose. The configuration mounts HuggingFace model caches to avoid re-downloading on each restart.

```bash
docker-compose -f docker/docker-compose.yaml up --build
```

Once running, access the dashboard at **http://localhost:8501**.

---

## ☁️ Production Cloud Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and link your repository.
3. Set your secrets in Streamlit Settings using the variable names from `.env.example`.

### AWS EC2
1. Spin up an Ubuntu EC2 instance (recommended: `t3.large` or above for model loading).
2. Install Docker and Docker Compose.
3. Clone the repository, write your `.env` file, and start the application:
   ```bash
   docker-compose -f docker/docker-compose.yaml up -d
   ```
4. Open port `8501` in the EC2 security group inbound rules.

---

## 📊 Evaluation Metrics

The unified evaluator runs both **RAGAS** and **DeepEval** metrics:

| Metric | Framework |
|---|---|
| Faithfulness | RAGAS |
| Context Precision | RAGAS |
| Context Recall | RAGAS |
| Answer Relevancy | RAGAS |
| Hallucination Score | DeepEval |
| Semantic Similarity | Custom (embedding-based) |

---

## 🎨 Usage Examples

### Query Routing
- **Greetings**: "Hi", "Hello bro" → General Chat Handler
- **General Knowledge**: "What is attention mechanism?" → General Knowledge Handler
- **Document Queries**: "What is the methodology used?" → CRAG RAG Pipeline

### CRAG Scenarios
- **GOOD Retrieval**: Query with clear document matches → Normal RAG pipeline
- **PARTIAL Retrieval**: Query with moderate relevance → Query rewrite + retry
- **BAD Retrieval**: Query with poor/no matches → Web search fallback

### Response Types
- **Document-Based**: High-quality retrieval with citations
- **General Knowledge**: Fallback when no documents available
- **Web Enhanced**: Document + web search context for poor retrieval

---

## 🔒 Security & Best Practices

- **API Keys**: Stored in `.env` file, never committed to version control
- **Connection Management**: Proper Weaviate connection closing to prevent resource leaks
- **Error Handling**: Comprehensive retry mechanisms and graceful fallbacks
- **Logging**: Structured logging with rotation for production monitoring
- **Input Validation**: Pydantic models for all data structures
- **Token Optimization**: Efficient prompt design and context compression

---

## 📈 Performance Optimizations

- **Citation Pipeline**: Removed LLM validation, using metadata extraction (saves ~200-500 tokens)
- **Response Length**: Increased max_tokens to 2048 for comprehensive answers
- **Context Compression**: Rule-based compression to 3000 characters
- **CRAG Loop**: Maximum 2 retrieval attempts to balance quality and latency
- **Web Search**: Limited to 3 results for token efficiency
- **Embedding Caching**: Local HuggingFace model cache

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request with a clear description

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **LangChain**: Framework for RAG orchestration
- **Groq**: Fast LLM inference
- **Weaviate**: Vector database
- **HuggingFace**: Embedding models
- **RAGAS & DeepEval**: Evaluation frameworks
