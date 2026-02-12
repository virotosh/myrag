"""
Pydantic schemas for chat-related API endpoints.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, validator, ConfigDict
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Message type enumeration."""
    TEXT = "text"
    FILE = "file"
    ERROR = "error"


class ConversationBase(BaseModel):
    """Base conversation schema."""
    title: Optional[str] = None


class ConversationCreate(ConversationBase):
    """Schema for creating a new conversation."""
    session_id: str


class ConversationUpdate(BaseModel):
    """Schema for updating conversation metadata."""
    title: Optional[str] = None
    is_active: Optional[bool] = None


class ConversationResponse(ConversationBase):
    """Schema for conversation API responses."""
    id: int
    session_id: str
    total_messages: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    
    # Pydantic v2 config (also disables protected namespace warning for 'model_used')
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class MessageBase(BaseModel):
    """Base message schema."""
    content: str
    role: MessageRole
    message_type: MessageType = MessageType.TEXT


class MessageCreate(MessageBase):
    """Schema for creating a new message."""
    conversation_id: int


class MessageUpdate(BaseModel):
    """Schema for updating message metadata."""
    sources_used: Optional[str] = None
    context_chunks: Optional[str] = None
    relevance_score: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[str] = None
    model_used: Optional[str] = None
    
    # Avoid conflict with protected namespace 'model_'
    model_config = ConfigDict(protected_namespaces=())

class DocumentMetadata(BaseModel):
    """Schema for source document references."""
    doi: str
    title: List[str]
    s2orcauthors: List[str]
    crossrefauthors: List[str]
    venue: str
    year: int
    url: str

class SourceDocument(BaseModel):
    """Schema for source document references."""
    document_id: int
    filename: str
    document_metadata: DocumentMetadata #str
    chunk_id: str
    relevance_score: float
    content_snippet: str

class MessageResponse(MessageBase):
    """Schema for message API responses."""
    id: int
    conversation_id: int
    sources_used: Optional[List[SourceDocument]] = None
    context_chunks: Optional[List[str]] = None
    relevance_score: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[str] = None
    model_used: Optional[str] = None
    created_at: datetime
    
    @validator('sources_used', pre=True)
    def parse_sources_used(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except:
                return None
        return v
    
    @validator('context_chunks', pre=True)
    def parse_context_chunks(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except:
                return None
        return v
    
    # Pydantic v2 config (also disables protected namespace warning for 'model_used')
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ChatRequest(BaseModel):
    """Schema for chat message requests."""
    message: str
    conversation_id: Optional[int] = None
    session_id: Optional[str] = None
    use_rag: bool = True
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    message: MessageResponse
    conversation: ConversationResponse
    processing_info: Dict[str, Any]


class ConversationListResponse(BaseModel):
    """Schema for paginated conversation list responses."""
    conversations: List[ConversationResponse]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool


class ChatHistoryResponse(BaseModel):
    """Schema for chat history responses."""
    conversation: ConversationResponse
    messages: List[MessageResponse]
    total_messages: int
