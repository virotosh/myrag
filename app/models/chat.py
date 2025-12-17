"""
Chat models for storing conversations and messages.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Conversation(Base):
    """Conversation model for storing chat sessions."""
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    session_id = Column(String(100), unique=True, index=True)
    
    # Metadata
    total_messages = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, session_id='{self.session_id}', messages={self.total_messages})>"


class Message(Base):
    """Message model for storing individual chat messages."""
    
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    
    # Message content
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    message_type = Column(String(20), default="text")  # text, file, error
    
    # RAG metadata
    sources_used = Column(Text, nullable=True)  # JSON string of source documents
    context_chunks = Column(Text, nullable=True)  # JSON string of retrieved chunks
    relevance_score = Column(String(10), nullable=True)  # High, Medium, Low
    
    # Processing metadata
    tokens_used = Column(Integer, nullable=True)
    processing_time = Column(String(20), nullable=True)  # in milliseconds
    model_used = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"
