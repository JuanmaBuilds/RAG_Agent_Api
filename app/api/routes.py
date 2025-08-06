from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
import logging
from typing import Optional, Dict, Any

from ..models.schemas import (
    QuestionRequest, 
    QuestionResponse, 
    DocumentRequest, 
    DocumentResponse,
    HealthResponse,
)
from ..services.rag_agent import rag_agent
from ..core.database import db_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Agent"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_connected = False
        try:
            await db_manager.connect()
            db_connected = True
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
        
        return HealthResponse(
            status="healthy",
            database_connected=db_connected,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG agent with enhanced search capabilities."""
    try:
        # Initialize RAG agent if needed
        await rag_agent.initialize()
        
        # Extract search filters from request if provided
        search_filters = None
        if hasattr(request, 'search_filters') and request.search_filters:
            search_filters = request.search_filters
        
        logger.info(f"Processing question: '{request.question}' with filters: {search_filters}")
        
        # Get answer from RAG agent
        result = await rag_agent.answer_question(
            question=request.question,
            context=request.context,
            max_tokens=request.max_tokens,
            search_filters=search_filters
        )
        
        # Log search metadata for debugging
        if "search_metadata" in result:
            logger.info(f"Search metadata: {result['search_metadata']}")
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a document to the vector database with chunking."""
    try:
        # Initialize RAG agent if needed
        await rag_agent.initialize()
        
        # Add document to RAG agent's vector store with chunking
        document_id = await rag_agent.add_document_to_chain(
            content=request.content,
            metadata=request.metadata
        )
        
        return DocumentResponse(
            document_id=document_id,
            status="success",
            message=f"Document added with chunking: {document_id}"
        )
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error adding document: {str(e)}"
        )


@router.post("/documents/reprocess")
async def reprocess_documents_with_chunking():
    """Re-process existing documents with chunking (experimental)."""
    try:
        await rag_agent.initialize()
        collection = rag_agent.vector_store_langchain.collection
        original_docs = collection.find({
            "$or": [
                {"metadata.chunk_index": {"$exists": False}},
                {"metadata.total_chunks": {"$exists": False}}
            ]
        })
        
        processed_count = 0
        for doc in original_docs:
            try:

                content = doc.get("search_text", doc.get("page_content", ""))
                metadata = doc.get("metadata", {})
                
                if content:
                    await rag_agent.add_document_to_chain(
                        content=content,
                        metadata=metadata
                    )
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error reprocessing document {doc.get('_id')}: {e}")
                continue
        
        return {
            "status": "success",
            "message": f"Reprocessed {processed_count} documents with chunking",
            "processed_count": processed_count
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reprocessing documents: {str(e)}"
        )


@router.put("/config/chunking")
async def update_chunking_config(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
):
    """Update chunking configuration."""
    try:
        from ..core.config import settings
        
        if chunk_size is not None:
            settings.chunk_size = chunk_size
        if chunk_overlap is not None:
            settings.chunk_overlap = chunk_overlap
        
        rag_agent._initialized = False
        await rag_agent.initialize()
        
        return {
            "status": "success",
            "message": "Chunking configuration updated",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        }
        
    except Exception as e:
        logger.error(f"Error updating chunking config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating chunking config: {str(e)}"
        )


@router.get("/config/chunking")
async def get_chunking_config():
    """Get current chunking configuration."""
    try:
        from ..core.config import settings
        
        return {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_separators": settings.chunk_separators
        }
        
    except Exception as e:
        logger.error(f"Error getting chunking config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chunking config: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    try:
        await rag_agent.initialize()
        
        results = await rag_agent.search_documents(
            query=f"document_id:{document_id}",
            limit=1
        )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return DocumentResponse(
            document_id=document_id,
            status="success",
            message="Document retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document: {str(e)}"
        )


@router.delete("/documents/{document_id}", response_model=DocumentResponse)
async def delete_document(document_id: str):
    """Delete a document from the vector database."""
    try:
        logger.info(f"Document deletion requested for ID: {document_id}")
        
        return DocumentResponse(
            document_id=document_id,
            status="success",
            message="Document deletion not yet implemented"
        )
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/search")
async def search_documents(
    query: str, 
    limit: int = Query(5, ge=1, le=50),
    search_type: str = Query("similarity", regex="^(similarity|vector|text)$"),
    document_type: Optional[str] = None,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """Search for documents in the vector database with enhanced filtering."""
    try:
        await rag_agent.initialize()
        
        filters = {}
        if document_type:
            filters["document_type"] = document_type
        if category:
            filters["category"] = category
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to
        
        logger.info(f"Searching documents with query: '{query}', type: {search_type}, filters: {filters}")
        
        results = await rag_agent.search_documents(query, limit, filters)
        
        return {
            "query": query,
            "search_type": search_type,
            "filters_applied": filters,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@router.get("/statistics")
async def get_search_statistics():
    """Get statistics about the vector store and search capabilities."""
    try:
        await rag_agent.initialize()
        
        stats = await rag_agent.get_search_statistics()
        
        return {
            "vector_store_statistics": stats,
            "search_capabilities": {
                "similarity_search": True,
                "vector_search": True,
                "text_search": True,
                "filtering": True,
                "langchain_integration": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )


@router.post("/test-search")
async def test_search_capabilities():
    """Test endpoint to verify search capabilities with sample queries."""
    try:
        await rag_agent.initialize()
        
        test_queries = [
            "equipaje permitido",
            "peso máximo maletas",
            "documentos de viaje",
            "check-in online"
        ]
        
        results = {}
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            
            similarity_results = await rag_agent.search_documents(query, 3)
            
            results[query] = {
                "similarity_results": len(similarity_results),
                "similarity_scores": [r.get("score", 0) for r in similarity_results[:3]]
            }
        
        return {
            "test_results": results,
            "summary": {
                "total_queries_tested": len(test_queries),
                "search_types_tested": ["similarity"],
                "status": "completed"
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing search capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error testing search capabilities: {str(e)}"
        ) 