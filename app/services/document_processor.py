"""
Document processing service for handling file uploads and text extraction.
"""
import os
import uuid
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import aiofiles
import logging
from datetime import datetime

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..core.config import settings
from ..models.document import Document as DocumentModel
from ..schemas.document import DocumentCreate, DocumentUpdate, ProcessingStatus

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing uploaded documents."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk and return the file path.
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename
            
        Returns:
            str: Path to saved file
        """
        # Generate unique filename to avoid conflicts
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.upload_dir / unique_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"File saved: {file_path}")
        return str(file_path)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text content from DOCX file."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text content from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            raise
    
    def extract_text_from_csv(self, file_path: str) -> str:
        """Extract text content from CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to text representation
            text = df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from CSV {file_path}: {str(e)}")
            raise
    
    def extract_text_from_file(self, file_path: str, file_type: str) -> str:
        """
        Extract text content from file based on file type.
        
        Args:
            file_path: Path to the file
            file_type: File extension (pdf, docx, txt, etc.)
            
        Returns:
            str: Extracted text content
        """
        file_type = file_type.lower().replace('.', '')
        
        extractors = {
            'pdf': self.extract_text_from_pdf,
            'docx': self.extract_text_from_docx,
            'doc': self.extract_text_from_docx,
            'txt': self.extract_text_from_txt,
            'md': self.extract_text_from_txt,
            'csv': self.extract_text_from_csv,
        }
        
        extractor = extractors.get(file_type)
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return extractor(file_path)
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        """
        Split text into chunks using LangChain text splitter.
        
        Args:
            text: Raw text content
            
        Returns:
            List[Document]: List of text chunks as LangChain Documents
        """
        try:
            documents = self.text_splitter.create_documents([text])
            logger.info(f"Text split into {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {str(e)}")
            raise
    
    async def process_document(
        self, 
        file_path: str, 
        document_model: DocumentModel
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, split into chunks.
        
        Args:
            file_path: Path to the uploaded file
            document_model: Database document model
            
        Returns:
            Dict containing processing results
        """
        try:
            logger.info(f"Starting processing for document {document_model.id}")
            
            # Extract text from file
            text_content = self.extract_text_from_file(
                file_path, 
                document_model.file_type
            )
            
            if not text_content.strip():
                raise ValueError("No text content extracted from file")
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text_content)
            
            # Calculate statistics
            word_count = len(text_content.split())
            chunk_count = len(chunks)
            content_preview = text_content[:500] + "..." if len(text_content) > 500 else text_content
            
            processing_result = {
                'text_content': text_content,
                'chunks': chunks,
                'word_count': word_count,
                'chunk_count': chunk_count,
                'content_preview': content_preview,
                'status': ProcessingStatus.COMPLETED
            }
            
            logger.info(f"Document {document_model.id} processed successfully: {chunk_count} chunks, {word_count} words")
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing document {document_model.id}: {str(e)}")
            return {
                'status': ProcessingStatus.FAILED,
                'error_message': str(e)
            }
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types."""
        return ['pdf', 'docx', 'doc', 'txt', 'md', 'csv']
    
    def validate_file_type(self, filename: str) -> bool:
        """Validate if file type is supported."""
        file_extension = Path(filename).suffix.lower().replace('.', '')
        return file_extension in self.get_supported_file_types()
    
    def validate_file_size(self, file_size: int) -> bool:
        """Validate if file size is within limits."""
        return file_size <= settings.max_file_size
    
    async def cleanup_file(self, file_path: str) -> None:
        """Remove file from disk."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")


# Global instance
document_processor = DocumentProcessor()
