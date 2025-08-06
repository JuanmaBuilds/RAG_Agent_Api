import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..core.config import settings
from ..core.database import db_manager

logger = logging.getLogger(__name__)


class RAGAgentService:
    """RAG (Retrieval-Augmented Generation) agent service using LangChain."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store_langchain = None
        self.qa_chain = None
        self.text_splitter = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the RAG agent with LangChain components."""
        try:
            # Initialize OpenAI components
            self.llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                temperature=0.1,
                max_tokens=1000
            )
            
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
            
            # Initialize text splitter for document chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=settings.chunk_separators,
                length_function=len,
            )
            
            # Initialize MongoDB Atlas vector store for LangChain
            await db_manager.connect()
            collection = db_manager.get_collection()
            
            self.vector_store_langchain = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name="default", 
                text_key="search_text",  
                embedding_key="embedding"
            )
            
            # Create domain-specific prompt template for airline information
            prompt_template = """Eres un asistente especializado en información de aerolíneas LATAM. Tu objetivo es proporcionar respuestas precisas y útiles basadas en la información disponible.

Contexto de información:
{context}

Pregunta del usuario: {question}

Instrucciones específicas:
1. Responde únicamente basándote en la información proporcionada en el contexto
2. Si la información no está disponible en el contexto, indícalo claramente.

Respuesta:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with improved configuration
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store_langchain.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 8,
                        "return_metadata": True, 
                        "include_scores": True
                    }
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            self._initialized = True
            logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {e}")
            raise
    
    async def answer_question(self, question: str, context: str = None, max_tokens: int = None, 
                            search_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Answer a question using the RAG system with LangChain's native retrieval."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            if max_tokens is not None and self.llm:
                self.llm.max_tokens = max_tokens
            
            full_question = question
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            
            if search_filters:
                logger.info(f"Using search filters: {search_filters}")
                # Create a new retriever with filters
                retriever = self.vector_store_langchain.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 8,
                        "filter": self._build_langchain_filter(search_filters)
                    }
                )
                
                # Create a new QA chain with the filtered retriever
                qa_chain_with_filters = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": self.qa_chain.combine_documents_chain.prompt},
                    return_source_documents=True
                )
                
                result = qa_chain_with_filters({"query": full_question})
            else:
                # Use standard RAG chain
                result = self.qa_chain({"query": full_question})
            
            answer = result.get("result", "I couldn't find a relevant answer to your question.")
            source_documents = result.get("source_documents", [])
            
            sources = []
            total_score = 0
            num_sources = len(source_documents)
            
            for doc in source_documents:
                if hasattr(doc, 'metadata') and 'document_id' in doc.metadata:
                    source_info = {
                        "document_id": doc.metadata['document_id'],
                        "score": doc.metadata.get('score', 0),
                        "vector_score": doc.metadata.get('vector_score', 0),
                        "text_score": doc.metadata.get('text_score', 0),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    sources.append(source_info)
                    total_score += doc.metadata.get('score', 0)
                elif hasattr(doc, 'page_content'):
                    # Extract a snippet from the content
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources.append({"content_preview": content})
            
            processing_time = time.time() - start_time
            
            response = {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time,
                "search_metadata": {
                    "num_sources": num_sources,
                    "total_score": total_score,
                    "avg_score": total_score / num_sources if num_sources > 0 else 0,
                    "search_filters_used": search_filters is not None
                }
            }
            
            logger.info(f"Question answered successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Error traceback: {traceback.format_exc()}")
            processing_time = time.time() - start_time
            
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def _build_langchain_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build LangChain-compatible filter from filters dictionary."""
        langchain_filter = {}
        
        for key, value in filters.items():
            if key == "document_type":
                langchain_filter["metadata.document_type"] = value
            elif key == "category":
                langchain_filter["metadata.category"] = value
            elif key == "date_from":
                langchain_filter["created_at"] = {"$gte": value}
            elif key == "date_to":
                if "created_at" in langchain_filter:
                    langchain_filter["created_at"]["$lte"] = value
                else:
                    langchain_filter["created_at"] = {"$lte": value}
            else:
                # Generic metadata filter
                langchain_filter[f"metadata.{key}"] = value
        
        return langchain_filter
    

    
    async def add_document_to_chain(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the LangChain vector store with chunking."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Create the original document
            original_doc = Document(page_content=content, metadata=metadata or {})
            
            # Split the document into chunks
            chunks = self.text_splitter.split_documents([original_doc])
            
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Add chunk metadata to each chunk and ensure search_text field
            for i, chunk in enumerate(chunks):
                # Generate a unique document ID for each chunk
                import uuid
                chunk_document_id = f"{metadata.get('document_id', 'doc')}_{i}_{uuid.uuid4().hex[:8]}" if metadata else f"chunk_{i}_{uuid.uuid4().hex[:8]}"
                
                # Update metadata with chunk info
                chunk.metadata.update({
                    "document_id": chunk_document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_document_id": metadata.get("document_id", "unknown") if metadata else "unknown",
                    "chunk_size": len(chunk.page_content)
                })
                
                # CRITICAL: Add search_text field that MongoDB Atlas vector search expects
                chunk.metadata["search_text"] = chunk.page_content
                
                # Also add the content to the root level for MongoDB Atlas compatibility
                chunk.page_content = chunk.page_content  # This ensures the content is preserved
            
            # Add chunks to LangChain vector store
            if self.vector_store_langchain and chunks:
                self.vector_store_langchain.add_documents(chunks)
            
            logger.info(f"Document added to RAG chain with {len(chunks)} chunks")
            return f"document_added_with_{len(chunks)}_chunks"
            
        except Exception as e:
            logger.error(f"Error adding document to RAG chain: {e}")
            raise
    
    async def search_documents(self, query: str, limit: int = 5, 
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using LangChain's retriever."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Create retriever with optional filters
            search_kwargs = {"k": limit}
            if filters:
                search_kwargs["filter"] = self._build_langchain_filter(filters)
            
            retriever = self.vector_store_langchain.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Convert to expected format
            results = []
            for doc in docs:
                result = {
                    "document_id": doc.metadata.get("document_id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search capabilities."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get collection statistics
            collection = self.vector_store_langchain.collection
            total_docs = collection.count_documents({})
            doc_types = list(collection.distinct("metadata.document_type"))
            categories = list(collection.distinct("metadata.category"))
            
            return {
                "total_documents": total_docs,
                "document_types": doc_types,
                "categories": categories,
                "indexes": list(collection.list_indexes())
            }
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {"error": str(e)}


# Global RAG agent service instance
rag_agent = RAGAgentService() 