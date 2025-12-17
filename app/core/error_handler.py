"""
Global error handling and logging utilities.
"""
import logging
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from .config import settings

logger = logging.getLogger(__name__)


def setup_error_logging():
    """Initialize error logging based on configuration."""
    if settings.error_logging == 2 and settings.sentry_dsn:
        # Initialize Sentry
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[
                FastApiIntegration(auto_enabling_integrations=False),
                SqlalchemyIntegration(),
            ],
            traces_sample_rate=0.1,
            environment="production" if not settings.debug else "development",
        )
        logger.info("Sentry error logging initialized")
    elif settings.error_logging == 1:
        # Setup file logging
        setup_file_logging()
        logger.info(f"File error logging initialized: {settings.error_log_dir}")
    else:
        logger.info("Error logging disabled")


def setup_file_logging():
    """Setup file-based error logging."""
    log_dir = Path(settings.error_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create error log file handler
    error_log_file = log_dir / "errors.log"
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    error_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(error_handler)


def log_error(
    error: Exception,
    request: Optional[Request] = None,
    extra_data: Optional[dict] = None
):
    """Log error to configured destination."""
    error_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    
    if request:
        error_data.update({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
        })
    
    if extra_data:
        error_data.update(extra_data)
    
    if settings.error_logging == 2 and settings.sentry_dsn:
        # Log to Sentry
        with sentry_sdk.push_scope() as scope:
            if request:
                scope.set_tag("method", request.method)
                scope.set_tag("url", str(request.url))
                scope.set_context("request", {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                })
            if extra_data:
                scope.set_context("extra", extra_data)
            sentry_sdk.capture_exception(error)
    
    elif settings.error_logging == 1:
        # Log to file
        log_file = Path(settings.error_log_dir) / "errors.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_data, indent=2) + "\n" + "-" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to write error to log file: {e}")
    
    # Always log to application logger
    logger.error(f"Error occurred: {error}", exc_info=True)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for FastAPI."""
    
    # Log the error
    log_error(exc, request)
    
    # Handle different types of exceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Exception",
                "detail": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    # For unhandled exceptions, return generic error in production
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc().split("\n")
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later."
            }
        )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation exceptions."""
    log_error(exc, request, {"error_type": "validation"})
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc)
        }
    )
