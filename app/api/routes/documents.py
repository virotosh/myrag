"""
API routes for document management (upload, list, delete, processing status).
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form
from sqlalchemy.orm import Session
import uuid
import os
from pathlib import Path

from ...core.database import get_db
from ...models.document import Document as DocumentModel
from ...schemas.document import (
    DocumentResponse, 
    DocumentListResponse, 
    DocumentUploadResponse,
    DocumentProcessingResponse,
    DocumentCreate,
    ProcessingStatus
)
from ...services.rag_service import rag_service
from ...services.document_processor import document_processor
from ..dependencies import validate_file_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    document_metadata: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document for processing.
    
    - **file**: Document file (PDF, DOCX, TXT, MD, CSV)
    
    Returns uploaded document info and starts background processing.
    """
    try:
        # Validate file
        file_content = await file.read()
        validate_file_upload(len(file_content), file.filename)
        
        # Save file to disk
        file_path = await document_processor.save_uploaded_file(
            file_content, file.filename
        )
        
        # Create document record
        document_data = DocumentCreate(
            filename=Path(file_path).name,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(file_content),
            file_type=Path(file.filename).suffix.lower(),
            mime_type=file.content_type
        )
        
        document_model = DocumentModel(**document_data.dict())
        db.add(document_model)
        db.commit()
        db.refresh(document_model)
        
        # Start background processing
        background_tasks.add_task(
            rag_service.process_document_pipeline,
            document_model,
            document_metadata,
            db
        )
        
        upload_id = str(uuid.uuid4())
        
        logger.info(f"Document uploaded: {document_model.id} - {file.filename}")
        
        return DocumentUploadResponse(
            message="Document uploaded successfully and processing started",
            document=DocumentResponse.from_orm(document_model),
            upload_id=upload_id
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    size: int = 20,
    status_filter: Optional[ProcessingStatus] = None,
    db: Session = Depends(get_db)
):
    """
    Get paginated list of uploaded documents.
    
    - **page**: Page number (starts from 1)
    - **size**: Number of documents per page
    - **status_filter**: Filter by processing status
    """
    try:
        # Build query
        query = db.query(DocumentModel)
        
        if status_filter:
            query = query.filter(DocumentModel.processing_status == status_filter)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * size
        documents = query.order_by(DocumentModel.created_at.desc())\
                        .offset(offset)\
                        .limit(size)\
                        .all()
        
        # Calculate pagination info
        has_next = offset + size < total
        has_prev = page > 1
        
        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            page=page,
            size=size,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get document details by ID."""
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return DocumentResponse.from_orm(document)


@router.get("/{document_id}/status", response_model=DocumentProcessingResponse)
async def get_document_processing_status(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get document processing status."""
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Calculate progress based on status
    progress = 0.0
    if document.processing_status == ProcessingStatus.PROCESSING:
        progress = 0.5
    elif document.processing_status == ProcessingStatus.COMPLETED:
        progress = 1.0
    elif document.processing_status == ProcessingStatus.FAILED:
        progress = 0.0
    
    return DocumentProcessingResponse(
        document_id=document.id,
        status=document.processing_status,
        progress=progress,
        message=document.error_message if document.processing_status == ProcessingStatus.FAILED else None,
        chunks_processed=document.chunk_count if document.is_processed else None,
        total_chunks=document.chunk_count if document.is_processed else None
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document and remove it from vector store."""
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Remove from RAG system (vector store + file system + database)
        success = await rag_service.delete_document_from_rag(document, db)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting document from RAG system"
            )
        
        logger.info(f"Document deleted: {document_id}")
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/reindex")
async def reindex_documents(
    background_tasks: BackgroundTasks,
    document_ids: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """
    Reindex documents in vector store.
    
    - **document_ids**: Optional list of specific document IDs to reindex
    """
    try:
        # Get documents to reindex
        query = db.query(DocumentModel)
        if document_ids:
            query = query.filter(DocumentModel.id.in_(document_ids))
        else:
            query = query.filter(DocumentModel.is_processed == True)
        
        documents = query.all()
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents found to reindex"
            )
        
        # Start background reindexing
        for document in documents:
            background_tasks.add_task(
                rag_service.process_document_pipeline,
                document,
                db
            )
        
        logger.info(f"Started reindexing {len(documents)} documents")
        
        return {
            "message": f"Started reindexing {len(documents)} documents",
            "document_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error starting reindex: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting reindex: {str(e)}"
        )


@router.get("/search/semantic")
async def search_documents(
    query: str,
    document_ids: Optional[List[int]] = None,
    max_results: int = 10
):
    """
    Perform semantic search across documents.
    
    - **query**: Search query
    - **document_ids**: Optional list of specific document IDs to search
    - **max_results**: Maximum number of results to return
    """
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        results = await rag_service.search_documents(
            query=query,
            document_ids=document_ids,
            max_results=max_results
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}"
        )
