"""
LLM service for generating responses using OpenAI GPT with RAG context.
"""
import logging
from typing import Dict, Any, Optional, List
import time
import json
from datetime import datetime

# OpenAI and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback

from ..core.config import settings
from .vector_store import vector_store_service

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating AI responses using OpenAI GPT with RAG."""
    
    def __init__(self):
        self.chat_model = ChatOpenAI(
            api_key=settings.openai_api_key,
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
        )
        
        # System prompt for RAG-enhanced responses
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided context from documents. 

Instructions:
1. Use the provided context to answer questions accurately and comprehensively
2. If the context doesn't contain enough information, say so clearly
3. Always cite which documents you're referencing when possible
4. Be concise but thorough in your responses
5. If no context is provided, answer based on your general knowledge but mention this limitation

Context will be provided in the following format:
[CONTEXT]
{context}
[/CONTEXT]

Always maintain a helpful and professional tone."""
        
        logger.info(f"LLM service initialized with {settings.llm_model}")
    
    async def generate_response(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        document_ids: Optional[List[int]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate AI response with optional RAG context.
        
        Args:
            user_query: User's question/message
            conversation_history: Previous messages in conversation
            use_rag: Whether to use RAG for context
            document_ids: Specific documents to search in
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Get RAG context if enabled
            context_info = {}
            if use_rag:
                context_info = await vector_store_service.get_relevant_context(
                    query=user_query,
                    max_chunks=50,
                    document_ids=document_ids
                )
            
            # Build messages for chat completion
            context_info.get('context_chunks', []) = context_info.get('context_chunks', [])[:5] # limit to 5 chunks used for context to RAG
            messages = self._build_messages(
                user_query=user_query,
                context_info=context_info,
                conversation_history=conversation_history
            )
            
            # Update model parameters if provided
            model_kwargs = {}
            if max_tokens:
                model_kwargs['max_completion_tokens'] = max_tokens
            if temperature is not None:
                model_kwargs['temperature'] = temperature
            
            # Generate response with token tracking
            with get_openai_callback() as cb:
                if model_kwargs:
                    # Create temporary model with custom parameters
                    temp_model = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model_name=settings.llm_model,
                        **model_kwargs
                    )
                    response = temp_model.invoke(messages)
                else:
                    response = self.chat_model.invoke(messages)
                
                tokens_used = cb.total_tokens
                cost = cb.total_cost
            
            processing_time = round((time.time() - start_time) * 1000)  # milliseconds
            
            # Determine relevance score based on context usage
            relevance_score = self._calculate_relevance_score(
                context_info.get('average_score', 0.0),
                context_info.get('total_chunks', 0)
            )
            
            result = {
                'response': response.content,
                'tokens_used': tokens_used,
                'processing_time': f"{processing_time}ms",
                'model_used': settings.llm_model,
                'cost': cost,
                'relevance_score': relevance_score,
                'context_info': context_info,
                'sources_used': context_info.get('source_documents', []),
                'context_chunks': context_info.get('context_chunks', [])#[:5] # showing that 5 chunks used for context to RAG
            }
            
            logger.info(f"Generated response in {processing_time}ms using {tokens_used} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def _build_messages(
        self,
        user_query: str,
        context_info: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List:
        """Build message list for chat completion."""
        messages = []
        
        # Add system message with context
        system_content = self.system_prompt
        if context_info.get('context_chunks'):
            context_text = "\n\n".join(context_info['context_chunks'])
            system_content += f"\n\n[CONTEXT]\n{context_text}\n[/CONTEXT]"
        
        messages.append(SystemMessage(content=system_content))
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Limit to last 10 messages
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
        
        # Add current user query
        messages.append(HumanMessage(content=user_query))
        
        return messages
    
    def _calculate_relevance_score(self, average_score: float, chunk_count: int) -> str:
        """Calculate relevance score category based on context quality."""
        if chunk_count == 0:
            return "None"
        elif average_score >= 0.77 and chunk_count >= 3:
            return "High"
        elif average_score >= 0.6 and chunk_count >= 2:
            return "Medium"
        else:
            return "Low"
    
    async def generate_conversation_title(self, first_message: str) -> str:
        """Generate a title for conversation based on first message."""
        try:
            prompt = f"""Generate a short, descriptive title (max 6 words) for a conversation that starts with this message:

"{first_message}"

Title:"""
            
            messages = [HumanMessage(content=prompt)]
            response = self.chat_model.invoke(messages)
            
            # Clean up the title
            title = response.content.strip().strip('"').strip("'")
            if len(title) > 50:
                title = title[:47] + "..."
            
            return title
            
        except Exception as e:
            logger.error(f"Error generating conversation title: {str(e)}")
            # Fallback to truncated first message
            return first_message[:30] + "..." if len(first_message) > 30 else first_message
    
    async def summarize_document_content(self, content: str, max_length: int = 200) -> str:
        """Generate a summary of document content."""
        try:
            prompt = f"""Summarize the following document content in {max_length} characters or less:

{content[:2000]}  # Limit input to avoid token limits

Summary:"""
            
            messages = [HumanMessage(content=prompt)]
            response = self.chat_model.invoke(messages)
            
            summary = response.content.strip()
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            # Fallback to truncated content
            return content[:max_length-3] + "..." if len(content) > max_length else content
    
    def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        try:
            # Simple health check with minimal token usage
            test_messages = [HumanMessage(content="Hello")]
            response = self.chat_model.invoke(test_messages)
            return bool(response.content)
        except Exception as e:
            logger.error(f"LLM service health check failed: {str(e)}")
            return False


# Global instance
llm_service = LLMService()
