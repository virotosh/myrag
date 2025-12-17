"""
Document model for storing uploaded files metadata.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.sql import func
from ..core.database import Base


class Document(Base):
    """Document model for storing file metadata and processing status."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100))
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Content metadata
    content_preview = Column(Text, nullable=True)
    document_metadata = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    
    # Vector store metadata
    vector_store_id = Column(String(100), nullable=True)
    embedding_model = Column(String(100), default="text-embedding-ada-002")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.processing_status}')>"
