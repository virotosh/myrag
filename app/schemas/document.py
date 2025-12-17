"""
Pydantic schemas for document-related API endpoints.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    """Base document schema with common fields."""
    filename: str
    file_type: str
    mime_type: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""
    original_filename: str
    file_path: str
    file_size: int


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata."""
    is_processed: Optional[bool] = None
    processing_status: Optional[ProcessingStatus] = None
    error_message: Optional[str] = None
    content_preview: Optional[str] = None
    document_metadata: Optional[str] = None
    chunk_count: Optional[int] = None
    word_count: Optional[int] = None
    vector_store_id: Optional[str] = None
    processed_at: Optional[datetime] = None


class DocumentResponse(DocumentBase):
    """Schema for document API responses."""
    id: int
    original_filename: str
    file_size: int
    is_processed: bool
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    content_preview: Optional[str] = None
    document_metadata: Optional[str] = None
    chunk_count: int
    word_count: int
    vector_store_id: Optional[str] = None
    embedding_model: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for paginated document list responses."""
    documents: List[DocumentResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool


class DocumentUploadResponse(BaseModel):
    """Schema for document upload responses."""
    message: str
    document: DocumentResponse
    upload_id: str


class DocumentProcessingResponse(BaseModel):
    """Schema for document processing status responses."""
    document_id: int
    status: ProcessingStatus
    progress: Optional[float] = None  # 0.0 to 1.0
    message: Optional[str] = None
    chunks_processed: Optional[int] = None
    total_chunks: Optional[int] = None
