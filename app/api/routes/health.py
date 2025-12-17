"""
Health check and system status API routes.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime

from ...core.database import get_db
from ...services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Chat RAG API"
    }


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check including all system components."""
    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_healthy = False
    
    # Check RAG system components
    rag_health = rag_service.health_check()
    
    # Overall system health
    overall_healthy = db_healthy and all(rag_health.values())
    
    health_status = {
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "healthy" if db_healthy else "unhealthy",
            "document_processor": "healthy" if rag_health.get('document_processor', False) else "unhealthy",
            "vector_store": "healthy" if rag_health.get('vector_store', False) else "unhealthy",
            "llm_service": "healthy" if rag_health.get('llm_service', False) else "unhealthy"
        }
    }
    
    if not overall_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics and metrics."""
    try:
        stats = await rag_service.get_rag_stats(db)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system stats: {str(e)}"
        )
