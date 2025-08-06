# RAG Agent API

FastAPI-based RAG (Retrieval-Augmented Generation) agent with LangChain, MongoDB Atlas, and OpenAI integration.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository-url>
cd rag_agent_api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file:

```env
# Required
MONGODB_ATLAS_URI=your_mongodb_atlas_connection_string
OPENAI_API_KEY=your_openai_api_key

# Optional - LangSmith for tracing and monitoring
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=

# Optional - App settings
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=true
```

### 3. MongoDB Atlas Setup

1. Create cluster at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Enable Vector Search with this index:
```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "search_text",
      "type": "filter"
    }
  ]
}
```

### 4. Run Application

```bash
python main.py
```

Access API docs: http://localhost:8000/docs

## 📚 API Usage

### Add Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document content here...",
    "metadata": {
      "title": "Document Title",
      "category": "Category"
    }
  }'
```

### Ask Question
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Your question here?",
    "max_tokens": 1000
  }'
```

### Search Documents
```bash
curl "http://localhost:8000/api/v1/search?query=search_term&limit=5"
```


## 🏗️ Project Structure

```
rag_agent_api/
├── app/
│   ├── api/routes.py          # API endpoints
│   ├── core/config.py         # Configuration
│   ├── core/database.py       # MongoDB connection
│   ├── services/rag_agent.py  # RAG logic
│   └── utils/logger.py        # Logging
├── main.py                    # FastAPI app
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## 🐳 Docker

```bash
docker build -t rag-agent-api .
docker run -p 8000:8000 --env-file .env rag-agent-api
```


```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test search
curl -X POST http://localhost:8000/api/v1/test-search
```

## 📖 API Documentation

- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
