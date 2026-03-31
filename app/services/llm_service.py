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
        temperature: Optional[float] = None,
        cached_context: Optional[Dict[str, Any]] = None,
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
            cached_context: Pre-built context dict (from a stored message).
                            When supplied, vector retrieval is skipped entirely
                            and this dict is used as-is for context_info.
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Get RAG context if enabled
            context_info = {}
            if cached_context is not None:
                # Reuse context that was already retrieved and stored for a
                # previous message – skip the vector store round-trip.
                context_info = cached_context
                logger.info(context_info['filters'])
                rerank = self.filter_documents(context_info['source_documents']+context_info['source_documents_notused'], context_info['filters'])
                context_info['source_documents'] = rerank[:5]
                context_info['source_documents_notused'] = rerank[5:]
                logger.info("FEEDBACK - Using cached context from stored message – skipping vector retrieval")
            elif use_rag:
                context_info = await vector_store_service.get_relevant_context(
                    query=user_query,
                    max_chunks=50,
                    chunks_used=5,
                    document_ids=document_ids
                )
            
            # Build messages for chat completion
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
            # summary
            summary_messages = self._build_messages_for_summary(
                user_query=user_query,
                context_info=context_info,
                rag_content=response.content
            )
            with get_openai_callback() as cb:
                if model_kwargs:
                    # Create temporary model with custom parameters
                    temp_model = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model_name=settings.llm_model,
                        **model_kwargs
                    )
                    summary = temp_model.invoke(summary_messages)
                else:
                    summary = self.chat_model.invoke(summary_messages)
            
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
                'sources_notused': context_info.get('source_documents_notused', []),
                'context_chunks': context_info.get('context_chunks', []),
                'context_chunks_notused': context_info.get('context_chunks_notused', []),
                'summary_included': summary.content
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

    def _build_messages_for_summary(
        self,
        user_query: str,
        context_info: Dict[str, Any],
        rag_content: str,
    ) -> List:
        messages = []
        summary_prompt = f"""
            Summary template:
            "This response draws on <document_sources> spanning from <years>. 
            These papers focus on a <specific theme>, published in <venues>. 
            Those <a list of all authors across document_sources> contribute <works>"

            Write a brief summary (200 words max, use summary template above) responds to the query "{user_query}"
            is based on the following souces, including inline 1-5 authors, important years, important venues, 
            and 5 most important keywords exists in "topics" key across all the sources to describe the theme in the summary:

            This is the response:
            {rag_content}

            Sources:
            {context_info['source_documents']}

        """
        logger.info(f"summary_prompt  {summary_prompt}")
        messages.append(HumanMessage(content=summary_prompt))
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



    ##### FILTERING LOGIC for FEEDBACK ##### 
    def normalize_name(self, name: str) -> str:
        """Lowercase and strip a name for fuzzy comparison."""
        return name.strip().lower()
 
 
    def author_match(self, doc_metadata: dict, filter_authors: list[str]) -> bool:
        """Return True if ALL filter_authors appear in either s2orcauthors or crossrefauthors."""
        s2_authors = {self.normalize_name(a) for a in doc_metadata.get("s2orcauthors", [])}
        cr_authors = {self.normalize_name(a) for a in doc_metadata.get("crossrefauthors", [])}
        combined_authors = s2_authors | cr_authors
     
        return all(self.normalize_name(fa) in combined_authors for fa in filter_authors)
     
     
    def topic_match(self, content_snippet: str, filter_topics: list[str]) -> bool:
        """Return True if ALL filter_topics appear (case-insensitive) in content_snippet."""
        snippet_lower = content_snippet.lower()
        return all(topic.lower() in snippet_lower for topic in filter_topics)
     
     
    def year_in_range(self, year: int, year_range: list[int]) -> bool:
        """Return True if year is within [year_range[0], year_range[1]]."""
        return year_range[0] <= year <= year_range[1]
     
     
    def filter_documents(
        self,
        documents: list[dict[str, Any]],
        filters: dict[str, dict],
    ) -> dict[str, list[dict]]:
        """
        Filter documents based on included/excluded criteria.
     
        Included: return documents matching ALL of authors, topics, and year range.
        Excluded: filter OUT documents that do NOT match ALL of authors, topics,
                  and year range (i.e. keep only those that fully satisfy all criteria).
     
        Parameters
        ----------
        documents : list of document dicts
        filters   : {
                        'included': {'authors': [...], 'topics': [...], 'years': [start, end]},
                        'excluded': {'authors': [...], 'topics': [...], 'years': [start, end]}
                    }
     
        Returns
        -------
        {'included': [...], 'excluded': [...]}
        """
        included_criteria = filters.get("included", {})
        excluded_criteria = filters.get("excluded", {})
     
        included_results = []
        excluded_results = []
     
        for doc in documents:
            meta = doc.get("document_metadata", {})
            snippet = doc.get("content_snippet", "")
            year = meta.get("year")
     
            # ── INCLUDED logic ──────────────────────────────────────────────────
            # Keep document only if it satisfies ALL three criteria.
            if included_criteria:
                inc_authors = included_criteria.get("authors", [])
                inc_topics = included_criteria.get("topics", [])
                inc_years = included_criteria.get("years", [])
     
                authors_ok = self.author_match(meta, inc_authors) if inc_authors else True
                topics_ok = self.topic_match(snippet, inc_topics) if inc_topics else True
                years_ok = self.year_in_range(year, inc_years) if (inc_years and year is not None) else True
     
                if authors_ok and topics_ok and years_ok:
                    included_results.append(doc)
     
            # ── EXCLUDED logic ───────────────────────────────────────────────────
            # Remove documents that fail ANY of the three criteria
            # (i.e. keep only those that satisfy ALL criteria).
            if excluded_criteria:
                exc_authors = excluded_criteria.get("authors", [])
                exc_topics = excluded_criteria.get("topics", [])
                exc_years = excluded_criteria.get("years", [])
     
                authors_ok = self.author_match(meta, exc_authors) if exc_authors else True
                authors_ok = authors_ok if len(exc_authors)>0 else False
                topics_ok = self.topic_match(snippet, exc_topics) if exc_topics else True
                topics_ok = topics_ok if len(exc_topics)>0 else False
                years_ok = self.year_in_range(year, exc_years) if (exc_years and year is not None) else True
                years_ok = years_ok if len(exc_years)>0 else False
     
                # Keep only documents that match ALL criteria (filter out non-matching ones)
                logger.info(authors_ok)
                logger.info(topics_ok)
                logger.info(years_ok)
                if not authors_ok and not topics_ok and not years_ok:
                    excluded_results.append(doc)
        matched = [ doc for doc in documents if doc in included_results and doc in excluded_results ]
        remaining = [ doc for doc in documents if doc not in matched ]
        logger.info("included_results")
        logger.info(included_results)
        logger.info("excluded_results")
        logger.info(excluded_results)
        return matched #+ remaining

# Global instance
llm_service = LLMService()
