"""
Main FastAPI application entry point.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
import uvicorn

from .core.config import settings
from .core.database import create_tables
from .core.error_handler import (
    setup_error_logging,
    global_exception_handler,
    validation_exception_handler
)
from .api.routes import documents, chat, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    description="RAG assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
if settings.debug:
    # In development, allow all hosts to prevent handshake issues (e.g., 127.0.0.1, LAN IPs)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
else:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
    )

# Setup error handling
setup_error_logging()

# Add exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Include API routes
app.include_router(health.router, prefix=settings.api_v1_str)
app.include_router(documents.router, prefix=settings.api_v1_str)
app.include_router(chat.router, prefix=settings.api_v1_str)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.project_name}")
    
    # Create database tables
    create_tables()
    logger.info("Database tables created/verified")
    
    # Log configuration
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Allowed origins: {settings.allowed_origins}")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Vector store directory: {settings.chroma_persist_directory}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info(f"Shutting down {settings.project_name}")


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Chat RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.api_v1_str}/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
