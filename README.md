# RAG Agent API

A FastAPI-based RAG (Retrieval-Augmented Generation) agent system that combines LangChain with MongoDB Atlas for intelligent question answering using vector embeddings.

## 🚀 Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **LangChain Integration**: Powerful framework for building LLM applications with native vector search
- **MongoDB Atlas Vector Search**: Scalable vector database for document embeddings
- **OpenAI Integration**: State-of-the-art language models for text generation and embeddings
- **RAG Architecture**: Retrieval-Augmented Generation for context-aware responses
- **Document Chunking**: Intelligent text splitting for better retrieval granularity
- **RESTful API**: Clean, documented API endpoints
- **Async Support**: High-performance asynchronous operations
- **Advanced Filtering**: Metadata-based filtering for precise search results
- **Confidence Scoring**: Enhanced confidence calculation based on multiple factors

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account with vector search enabled
- OpenAI API key
- pip (Python package manager)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag_agent_api
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` file with your configuration:
   ```env
   # MongoDB Atlas Configuration
   MONGODB_ATLAS_URI=your_mongodb_atlas_connection_string
   MONGODB_DATABASE_NAME=rag_database
   MONGODB_COLLECTION_NAME=documents
   
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key
   
   # Application Configuration
   APP_HOST=0.0.0.0
   APP_PORT=8000
   APP_DEBUG=True
   
   # Vector Database Configuration
   VECTOR_DIMENSION=1536
   EMBEDDING_MODEL=text-embedding-ada-002
   ```

## 🗄️ MongoDB Atlas Setup

1. **Create a MongoDB Atlas cluster**:
   - Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
   - Create a new cluster (M0 free tier is sufficient for testing)

2. **Enable Vector Search**:
   - In your cluster, go to "Search" tab
   - Create a search index with the following configuration:
   ```json
   {
     "mappings": {
       "dynamic": true,
       "fields": {
         "embedding": {
           "dimensions": 1536,
           "similarity": "cosine",
           "type": "knnVector"
         }
       }
     }
   }
   ```

3. **Get your connection string**:
   - Go to "Database Access" and create a user
   - Go to "Network Access" and add your IP (or 0.0.0.0/0 for all)
   - Click "Connect" and copy the connection string

## 🚀 Running the Application

1. **Start the server**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/api/v1/health

## 📚 API Endpoints

### Health Check
```http
GET /api/v1/health
```

### Ask a Question
```http
POST /api/v1/ask
Content-Type: application/json

{
  "question": "What is the capital of France?",
  "context": "Additional context if needed",
  "max_tokens": 1000,
  "search_filters": {
    "document_type": "geography",
    "category": "capitals"
  }
}
```

### Add Document
```http
POST /api/v1/documents
Content-Type: application/json

{
  "content": "Document content to be vectorized and chunked",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "category": "Category",
    "document_type": "type"
  },
  "document_id": "optional-custom-id"
}
```

### Configure Chunking
```http
GET /api/v1/config/chunking
```

```http
PUT /api/v1/config/chunking?chunk_size=1000&chunk_overlap=200
```

### Re-process Existing Documents
```http
POST /api/v1/documents/reprocess
```

### Get Document
```http
GET /api/v1/documents/{document_id}
```

### Delete Document
```http
DELETE /api/v1/documents/{document_id}
```

### Search Documents
```http
GET /api/v1/search?query=search_term&limit=5&document_type=type&category=category
```

### Get Statistics
```http
GET /api/v1/statistics
```

### Test Search Capabilities
```http
POST /api/v1/test-search
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_ATLAS_URI` | MongoDB Atlas connection string | Required |
| `MONGODB_DATABASE_NAME` | Database name | `rag_database` |
| `MONGODB_COLLECTION_NAME` | Collection name | `documents` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `APP_HOST` | Application host | `0.0.0.0` |
| `APP_PORT` | Application port | `8000` |
| `APP_DEBUG` | Debug mode | `True` |
| `VECTOR_DIMENSION` | Embedding dimension | `1536` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `CHUNK_SIZE` | Document chunk size in characters | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks in characters | `200` |

### Chunking Configuration

The system now supports intelligent document chunking to improve retrieval performance:

- **Chunk Size**: Default 1000 characters per chunk
- **Chunk Overlap**: Default 200 characters overlap between chunks
- **Separators**: Intelligent splitting on paragraphs, sentences, and words
- **Metadata Preservation**: Each chunk retains original document metadata plus chunk-specific info

Benefits of chunking:
- ✅ Better retrieval granularity
- ✅ More precise context matching
- ✅ Improved performance for large documents
- ✅ Better handling of long-form content

## 📁 Project Structure

```
rag_agent_api/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   └── database.py        # MongoDB connection
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── rag_agent.py       # RAG agent service with LangChain integration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py          # Logging utilities
│   └── __init__.py
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── env.example               # Environment variables template
└── README.md                 # This file
```

## 🔍 Usage Examples

### Using curl

1. **Add a document**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents" \
        -H "Content-Type: application/json" \
        -d '{
          "content": "Paris is the capital and largest city of France. It is known as the City of Light and is famous for its art, fashion, gastronomy and culture.",
          "metadata": {
            "title": "Paris Information",
            "category": "Geography",
            "document_type": "geography"
          }
        }'
   ```

2. **Ask a question with filters**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/ask" \
        -H "Content-Type: application/json" \
        -d '{
          "question": "What is the capital of France?",
          "search_filters": {
            "document_type": "geography",
            "category": "capitals"
          },
          "max_tokens": 500
        }'
   ```

3. **Search documents with filters**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/search?query=capital&document_type=geography&limit=5"
   ```

### Using Python requests

```python
import requests

# Add document
response = requests.post(
    "http://localhost:8000/api/v1/documents",
    json={
        "content": "Your document content here...",
        "metadata": {
            "title": "Document Title",
            "category": "Category",
            "document_type": "type"
        }
    }
)

# Ask question with filters
response = requests.post(
    "http://localhost:8000/api/v1/ask",
    json={
        "question": "Your question here?",
        "search_filters": {
            "document_type": "type",
            "category": "category"
        },
        "max_tokens": 1000
    }
)

print(response.json())
```

## 🧪 Testing

1. **Health check**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. **Test search capabilities**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/test-search
   ```

3. **Run the test script**:
   ```bash
   ./test_curl_commands.sh
   ```

## 🔒 Security Considerations

- **Environment Variables**: Never commit your `.env` file to version control
- **API Keys**: Keep your OpenAI API key secure
- **Database Access**: Use proper authentication for MongoDB Atlas
- **CORS**: Configure CORS properly for production use
- **Rate Limiting**: Consider implementing rate limiting for production

## 🚀 Deployment

### Docker (Recommended)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8000
   
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**:
   ```bash
   docker build -t rag-agent-api .
   docker run -p 8000:8000 --env-file .env rag-agent-api
   ```

### Production Considerations

- Use a production WSGI server like Gunicorn
- Set up proper logging and monitoring
- Configure environment variables securely
- Set up SSL/TLS certificates
- Implement proper error handling and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**:
   - Verify your connection string
   - Check network access settings
   - Ensure vector search is enabled

2. **OpenAI API Error**:
   - Verify your API key
   - Check API quota and billing

3. **Vector Search Not Working**:
   - Ensure the search index is created correctly
   - Check the index name matches your configuration

4. **LangChain Integration Issues**:
   - Verify LangChain version compatibility
   - Check MongoDB Atlas vector search configuration
   - Ensure proper embedding model configuration

### Logs

Check the application logs for detailed error information:
```bash
tail -f app.log
```

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Open an issue on the repository 