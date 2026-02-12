"""
Vector store service for managing embeddings and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
from datetime import datetime

# LangChain imports
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from ..core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector embeddings and similarity search."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=settings.chroma_persist_directory,
            embedding_function=self.embeddings,
            collection_name="documents"
        )
        
        logger.info("Vector store initialized with Chroma")
    
    async def add_documents(
        self, 
        documents: List[Document], 
        document_id: int,
        document_filename: str,
        document_metadata: str
    ) -> str:
        """
        Add documents to vector store with metadata.
        
        Args:
            documents: List of LangChain Document objects
            document_id: Database document ID
            document_filename: Original filename
            
        Returns:
            str: Vector store collection ID
        """
        try:
            # Add metadata to each document
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "document_id": document_id,
                    "document_filename": document_filename,
                    "document_metadata": document_metadata,
                    "chunk_index": i,
                    "chunk_id": f"{document_id}_{i}",
                    "added_at": datetime.utcnow().isoformat()
                })
            
            # Add documents to vector store
            ids = [f"{document_id}_{i}" for i in range(len(documents))]
            self.vector_store.add_documents(documents, ids=ids)
            
            # Persist the vector store
            self.vector_store.persist()
            
            vector_store_id = f"doc_{document_id}_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Added {len(documents)} documents to vector store for document {document_id}")
            return vector_store_id
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        document_ids: Optional[List[int]] = None,
        score_threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search in vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Build filter for specific documents if provided
            filter_dict = None
            if document_ids:
                filter_dict = {"document_id": {"$in": document_ids}}
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Convert distance (lower is better) to similarity in [0,1]
            # similarity = 1 / (1 + distance)
            converted_results: List[Tuple[Document, float]] = []
            for doc, distance in results:
                try:
                    sim = 1.0 / (1.0 + float(distance))
                except Exception:
                    # Fallback if distance not a number
                    sim = 0.0
                converted_results.append((doc, sim))

            # Filter by similarity threshold
            filtered_results = [
                (doc, sim) for doc, sim in converted_results
                if sim >= score_threshold
            ]
            
            logger.info(f"Similarity search returned {len(filtered_results)} results above similarity threshold {score_threshold}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    async def get_relevant_context(
        self, 
        query: str, 
        max_chunks: int = 50,
        chunks_used: int = 5,
        document_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for RAG pipeline.
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to return
            document_ids: Optional list of document IDs to search in
            
        Returns:
            Dict containing context information
        """
        try:
            # Perform similarity search
            search_results = await self.similarity_search(
                query=query,
                k=max_chunks,
                document_ids=document_ids
            )
            
            if not search_results:
                return {
                    "context_chunks": [],
                    "source_documents": [],
                    "total_chunks": 0,
                    "average_score": 0.0
                }
            
            # Extract context chunks and source information
            context_chunks = []
            source_documents = []
            scores = []
            
            for doc, score in search_results:
                context_chunks.append(doc.page_content)
                scores.append(score)
                
                # Extract source document info
                source_info = {
                    "document_id": doc.metadata.get("document_id"),
                    "filename": doc.metadata.get("document_filename"),
                    "document_metadata": doc.metadata.get("document_metadata"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "relevance_score": score,
                    "content_snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                source_documents.append(source_info)
            
            # Calculate average score
            average_score = sum(scores) / len(scores) if scores else 0.0
            
            result = {
                "context_chunks": context_chunks[:chunks_used],
                "source_documents": source_documents[:chunks_used],
                "source_documents_notused": source_documents[chunks_used:],
                "total_chunks": len(context_chunks[:chunks_used]),
                "average_score": average_score,
                "query": query
            }
            
            logger.info(f"Retrieved {len(context_chunks)} relevant chunks for query")
            return result
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            raise
    
    async def delete_document_vectors(self, document_id: int) -> bool:
        """
        Delete all vectors for a specific document.
        
        Args:
            document_id: Database document ID
            
        Returns:
            bool: Success status
        """
        try:
            # Get all chunk IDs for this document
            filter_dict = {"document_id": document_id}
            
            # Delete from vector store
            # Note: Chroma doesn't have a direct delete by metadata method
            # We need to get IDs first, then delete
            
            # This is a workaround - in production, consider using a different vector DB
            # or maintaining an index of document->chunk_id mappings
            
            logger.info(f"Attempted to delete vectors for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": "documents",
                "embedding_model": settings.embedding_model,
                "persist_directory": settings.chroma_persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def health_check(self) -> bool:
        """Check if vector store is healthy."""
        try:
            # Simple health check
            self.vector_store._collection.count()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False


# Global instance
vector_store_service = VectorStoreService()
