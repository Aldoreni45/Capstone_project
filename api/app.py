from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import tempfile
import os
from pathlib import Path

from custom_logging.logger import app_logger
from chains.rag_pipeline import RAGPipeline
from config.settings import settings
from loaders.pdf_loader import ResearchPaperLoader
from chunking.recursive import RecursiveChunker
from embeddings import get_embeddings
from vectordb.weaviate_client import WeaviateVectorClient

# Initialize FastAPI app
app = FastAPI(
    title="CRAG RAG API",
    description="Production-grade CRAG (Corrective RAG) API for research paper Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query to answer", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    mode: str = Field("rag", description="Query mode: 'rag', 'general_chat', or 'hybrid'")
    memory_type: str = Field("buffer", description="Memory type: 'buffer', 'summary', or 'token'")


class Citation(BaseModel):
    """Citation model for response."""
    chunk_id: str
    title: str
    source: str
    page: int
    passage: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    citations: List[Citation]
    confidence: float
    retrieval_quality: str
    retrieval_reason: str
    query_rewritten: bool
    web_search_used: bool
    session_id: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    pipeline_ready: bool
    message: str


class IngestionResponse(BaseModel):
    """Response model for PDF ingestion endpoint."""
    status: str
    document_id: str
    filename: str
    chunks_created: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global rag_pipeline
    try:
        app_logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        app_logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        app_logger.error(f"Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline_ready = rag_pipeline is not None
    status = "healthy" if pipeline_ready else "unhealthy"
    message = "RAG pipeline is ready" if pipeline_ready else "RAG pipeline not initialized"
    
    return HealthResponse(
        status=status,
        pipeline_ready=pipeline_ready,
        message=message
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query endpoint for RAG pipeline.
    
    Args:
        request: QueryRequest containing query, session_id, mode, and memory_type
        
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        app_logger.info(f"Received query: {request.query}")
        
        # Generate session_id if not provided
        session_id = request.session_id or f"sess_{hash(request.query)}"
        
        # Execute RAG pipeline
        embedding_model_type = settings.get("embeddings", "default_model", default="bge")
        result = rag_pipeline.answer_query(
            query=request.query,
            session_id=session_id,
            embedding_model_type=embedding_model_type,
            memory_type=request.memory_type
        )
        
        # Handle tuple result format (StructuredAnswer, chunks, metrics)
        if isinstance(result, tuple) and len(result) == 3:
            ans_struct, chunks, metrics = result
            
            # Extract answer and metadata
            answer = ans_struct.answer
            confidence = ans_struct.confidence_score
            
            # Convert citations from StructuredAnswer
            citations = []
            for cite in ans_struct.citations:
                citations.append(Citation(
                    chunk_id=getattr(cite, 'chunk_id', ''),
                    title=getattr(cite, 'title', ''),
                    source=getattr(cite, 'source', ''),
                    page=getattr(cite, 'page', 0),
                    passage=getattr(cite, 'passage', '')
                ))
            
            # Extract metrics
            retrieval_quality = metrics.get("retrieval_quality", "UNKNOWN")
            retrieval_reason = metrics.get("retrieval_reason", "")
            query_rewritten = metrics.get("query_rewritten", False)
            web_search_used = metrics.get("web_search_used", False)
        elif isinstance(result, dict):
            # Handle dict format (BAD retrieval case)
            answer = result.get("answer", "")
            citations_data = result.get("citations", [])
            confidence = result.get("confidence", 0.0)
            retrieval_quality = result.get("retrieval_quality", "UNKNOWN")
            retrieval_reason = result.get("retrieval_reason", "")
            query_rewritten = result.get("query_rewritten", False)
            web_search_used = result.get("web_search_used", False)
            
            # Convert citations to Pydantic models
            citations = []
            for cite in citations_data:
                if isinstance(cite, dict):
                    citations.append(Citation(**cite))
                else:
                    citations.append(Citation(
                        chunk_id=getattr(cite, 'chunk_id', ''),
                        title=getattr(cite, 'title', ''),
                        source=getattr(cite, 'source', ''),
                        page=getattr(cite, 'page', 0),
                        passage=getattr(cite, 'passage', '')
                    ))
        else:
            raise HTTPException(status_code=500, detail="Unexpected result format from RAG pipeline")
        
        app_logger.info(f"Query completed successfully. Quality: {retrieval_quality}")
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieval_quality=retrieval_quality,
            retrieval_reason=retrieval_reason,
            query_rewritten=query_rewritten,
            web_search_used=web_search_used,
            session_id=session_id
        )
        
    except Exception as e:
        app_logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF file into the vector database.
    
    This endpoint:
    1. Accepts a PDF file upload
    2. Loads and extracts metadata from the PDF
    3. Chunks the document
    4. Embeds chunks
    5. Stores in Weaviate vector database
    
    Args:
        file: PDF file to ingest
        
    Returns:
        IngestionResponse with status and metadata
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        app_logger.info(f"Processing PDF ingestion: {file.filename}")
        
        # Step 1: Load PDF with metadata extraction
        loader = ResearchPaperLoader()
        documents = loader.load_paper(temp_file.name)
        app_logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Step 2: Chunk documents
        chunker = RecursiveChunker()
        chunks = chunker.split_documents(documents)
        app_logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Initialize embedding client
        embedding_client = get_embeddings(settings.get("embeddings", "default_model", default="huggingface"))
        
        # Step 4: Store in Weaviate
        weaviate_client = WeaviateVectorClient()
        weaviate_client.connect()
        
        # Generate document ID
        document_id = f"doc_{hash(file.filename)}"
        
        # Store chunks in Weaviate
        stored_count = 0
        for chunk in chunks:
            try:
                # Embed chunk
                embedding = embedding_client.embed_query(chunk.page_content)
                
                # Store in Weaviate
                weaviate_client.store_chunk(
                    chunk_id=f"{document_id}_chunk_{stored_count}",
                    content=chunk.page_content,
                    embedding=embedding,
                    metadata={
                        "document_id": document_id,
                        "filename": file.filename,
                        "paper_title": chunk.metadata.get("paper_title", ""),
                        "author": chunk.metadata.get("author", ""),
                        "page_number": chunk.metadata.get("page_number", 0),
                        "source": chunk.metadata.get("source", file.filename),
                        "chunk_number": chunk.metadata.get("chunk_number", 0)
                    }
                )
                stored_count += 1
            except Exception as e:
                app_logger.error(f"Error storing chunk {stored_count}: {e}")
                continue
        
        weaviate_client.close()
        
        app_logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks")
        
        return IngestionResponse(
            status="success",
            document_id=document_id,
            filename=file.filename,
            chunks_created=stored_count,
            message=f"Successfully ingested PDF with {stored_count} chunks"
        )
        
    except Exception as e:
        app_logger.error(f"Error during PDF ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting PDF: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CRAG RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
