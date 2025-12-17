"""
API dependencies for FastAPI routes.
"""
from typing import Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.config import settings


def get_database() -> Generator[Session, None, None]:
    """Get database session dependency."""
    return get_db()


def validate_file_upload(file_size: int, filename: str) -> None:
    """Validate uploaded file parameters."""
    from ..services.document_processor import document_processor
    
    if not document_processor.validate_file_size(file_size):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size // (1024 * 1024)}MB"
        )
    
    if not document_processor.validate_file_type(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Supported types: {document_processor.get_supported_file_types()}"
        )
