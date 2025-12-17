"""
RAG (Retrieval-Augmented Generation) service that orchestrates document processing,
vector storage, and LLM response generation.
"""
import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from sqlalchemy.orm import Session
from ..models.document import Document as DocumentModel
from ..models.chat import Conversation, Message
from ..schemas.document import DocumentUpdate, ProcessingStatus
from ..schemas.chat import MessageRole, MessageType

from .document_processor import document_processor
from .vector_store import vector_store_service
from .llm_service import llm_service

logger = logging.getLogger(__name__)


class RAGService:
    """Main service that orchestrates the RAG pipeline."""
    
    def __init__(self):
        self.document_processor = document_processor
        self.vector_store = vector_store_service
        self.llm_service = llm_service
        logger.info("RAG service initialized")
    
    async def process_document_pipeline(
        self, 
        document_model: DocumentModel,
        document_metadata: str, 
        db: Session
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline: extract text, create embeddings, store in vector DB.
        
        Args:
            document_model: Database document model
            db: Database session
            
        Returns:
            Dict containing processing results
        """
        try:
            logger.info(f"Starting RAG pipeline for document {document_model.id}")
            
            # Update status to processing
            document_model.processing_status = ProcessingStatus.PROCESSING
            db.commit()
            
            # Step 1: Process document (extract text, split into chunks)
            processing_result = await self.document_processor.process_document(
                document_model.file_path,
                document_model
            )
            
            if processing_result['status'] == ProcessingStatus.FAILED:
                # Update document with error
                update_data = DocumentUpdate(
                    processing_status=ProcessingStatus.FAILED,
                    error_message=processing_result.get('error_message')
                )
                self._update_document_model(document_model, update_data, db)
                return processing_result
            
            # Step 2: Add to vector store
            vector_store_id = await self.vector_store.add_documents(
                documents=processing_result['chunks'],
                document_id=document_model.id,
                document_filename=document_model.original_filename,
                document_metadata=document_metadata
            )
            
            # Step 3: Update document model with results
            update_data = DocumentUpdate(
                is_processed=True,
                processing_status=ProcessingStatus.COMPLETED,
                content_preview=processing_result['content_preview'],
                document_metadata=document_metadata,
                chunk_count=processing_result['chunk_count'],
                word_count=processing_result['word_count'],
                vector_store_id=vector_store_id,
                processed_at=datetime.utcnow()
            )
            
            self._update_document_model(document_model, update_data, db)
            
            result = {
                'status': ProcessingStatus.COMPLETED,
                'document_id': document_model.id,
                'vector_store_id': vector_store_id,
                'chunk_count': processing_result['chunk_count'],
                'word_count': processing_result['word_count']
            }
            
            logger.info(f"RAG pipeline completed for document {document_model.id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline for document {document_model.id}: {str(e)}")
            
            # Update document with error
            update_data = DocumentUpdate(
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
            self._update_document_model(document_model, update_data, db)
            
            return {
                'status': ProcessingStatus.FAILED,
                'error_message': str(e)
            }
    
    async def generate_rag_response(
        self,
        user_query: str,
        conversation_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        db: Optional[Session] = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        Generate RAG-enhanced response to user query.
        
        Args:
            user_query: User's question
            conversation_id: Optional conversation ID for history
            document_ids: Optional specific documents to search
            db: Database session for conversation history
            **llm_kwargs: Additional arguments for LLM
            
        Returns:
            Dict containing response and metadata
        """
        try:
            logger.info(f"Generating RAG response for query: {user_query[:50]}...")
            
            # Get conversation history if available
            conversation_history = []
            if conversation_id and db:
                conversation_history = self._get_conversation_history(conversation_id, db)
            
            # Generate response using LLM service with RAG
            response_data = await self.llm_service.generate_response(
                user_query=user_query,
                conversation_history=conversation_history,
                use_rag=True,
                document_ids=document_ids,
                **llm_kwargs
            )
            logger.info(f"AI response: {response_data}")
            logger.info(f"RAG response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            raise
    
    async def search_documents(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search documents using vector similarity.
        
        Args:
            query: Search query
            document_ids: Optional specific documents to search
            max_results: Maximum number of results
            
        Returns:
            Dict containing search results
        """
        try:
            search_results = await self.vector_store.similarity_search(
                query=query,
                k=max_results,
                document_ids=document_ids
            )
            
            # Format results
            formatted_results = []
            for doc, score in search_results:
                result = {
                    'document_id': doc.metadata.get('document_id'),
                    'filename': doc.metadata.get('document_filename'),
                    'chunk_id': doc.metadata.get('chunk_id'),
                    'content': doc.page_content,
                    'relevance_score': score,
                    'chunk_index': doc.metadata.get('chunk_index')
                }
                formatted_results.append(result)
            
            return {
                'results': formatted_results,
                'total_results': len(formatted_results),
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    async def delete_document_from_rag(
        self,
        document_model: DocumentModel,
        db: Session
    ) -> bool:
        """
        Remove document from RAG system (vector store and file system).
        
        Args:
            document_model: Document to remove
            db: Database session
            
        Returns:
            bool: Success status
        """
        try:
            # Remove from vector store
            await self.vector_store.delete_document_vectors(document_model.id)
            
            # Clean up file
            await self.document_processor.cleanup_file(document_model.file_path)
            
            # Remove from database
            db.delete(document_model)
            db.commit()
            
            logger.info(f"Document {document_model.id} removed from RAG system")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document from RAG: {str(e)}")
            return False
    
    def _update_document_model(
        self, 
        document_model: DocumentModel, 
        update_data: DocumentUpdate, 
        db: Session
    ) -> None:
        """Update document model with new data."""
        for field, value in update_data.dict(exclude_unset=True).items():
            setattr(document_model, field, value)
        db.commit()
        db.refresh(document_model)
    
    def _get_conversation_history(
        self, 
        conversation_id: int, 
        db: Session,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        try:
            messages = db.query(Message)\
                .filter(Message.conversation_id == conversation_id)\
                .order_by(Message.created_at.desc())\
                .limit(limit)\
                .all()
            
            # Reverse to get chronological order
            messages.reverse()
            
            history = []
            for msg in messages:
                history.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    async def get_rag_stats(self, db: Session) -> Dict[str, Any]:
        """Get RAG system statistics."""
        try:
            # Document stats
            total_docs = db.query(DocumentModel).count()
            processed_docs = db.query(DocumentModel)\
                .filter(DocumentModel.is_processed == True).count()
            failed_docs = db.query(DocumentModel)\
                .filter(DocumentModel.processing_status == ProcessingStatus.FAILED).count()
            
            # Vector store stats
            vector_stats = await self.vector_store.get_collection_stats()
            
            # Conversation stats
            total_conversations = db.query(Conversation).count()
            total_messages = db.query(Message).count()
            
            return {
                'documents': {
                    'total': total_docs,
                    'processed': processed_docs,
                    'failed': failed_docs,
                    'processing_rate': round(processed_docs / total_docs * 100, 2) if total_docs > 0 else 0
                },
                'vector_store': vector_stats,
                'conversations': {
                    'total': total_conversations,
                    'total_messages': total_messages,
                    'avg_messages_per_conversation': round(total_messages / total_conversations, 2) if total_conversations > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG stats: {str(e)}")
            return {}
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all RAG components."""
        return {
            'document_processor': True,  # No specific health check needed
            'vector_store': self.vector_store.health_check(),
            'llm_service': self.llm_service.health_check()
        }


# Global instance
rag_service = RAGService()
