import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import db_manager
from app.services.rag_agent import rag_agent
from app.api.routes import router
from app.utils.logger import setup_logging


# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Agent API...")
    
    try:
        # Initialize database connection
        await db_manager.connect()
        logger.info("Database connection established")
        
        # Initialize RAG agent (which now handles vector store initialization)
        await rag_agent.initialize()
        logger.info("RAG agent initialized")
        
        logger.info("RAG Agent API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Agent API...")
    
    try:
        await db_manager.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="RAG Agent API",
    description="A FastAPI-based RAG (Retrieval-Augmented Generation) agent system using LangChain and MongoDB Atlas",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG Agent"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "description": "A FastAPI-based RAG agent system using LangChain and MongoDB Atlas",
        "endpoints": {
            "health": "/api/v1/health",
            "ask": "/api/v1/ask",
            "documents": "/api/v1/documents",
            "search": "/api/v1/search"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info"
    ) 