"""
API routes for chat functionality (send messages, get history, WebSocket).
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import json
import uuid
from datetime import datetime

from ...core.database import get_db
from ...models.chat import Conversation, Message
from ...schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationListResponse,
    ChatHistoryResponse,
    MessageResponse,
    MessageRole,
    MessageType,
    ConversationCreate,
    MessageCreate
)
from ...services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ConnectionManager:
    """WebSocket connection manager for real-time chat."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_connections: dict = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        if websocket not in self.active_connections:
            self.active_connections.append(websocket)
        if conversation_id not in self.conversation_connections:
            self.conversation_connections[conversation_id] = []
        if websocket not in self.conversation_connections[conversation_id]:
            self.conversation_connections[conversation_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, conversation_id: str):
        # Safely remove from active connections
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        except ValueError:
            pass
        # Safely remove from conversation mapping
        if conversation_id in self.conversation_connections:
            try:
                if websocket in self.conversation_connections[conversation_id]:
                    self.conversation_connections[conversation_id].remove(websocket)
                # Clean up empty lists
                if not self.conversation_connections[conversation_id]:
                    del self.conversation_connections[conversation_id]
            except ValueError:
                pass
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def send_to_conversation(self, message: str, conversation_id: str):
        if conversation_id in self.conversation_connections:
            # Iterate over a copy to allow safe mutation during iteration
            for connection in list(self.conversation_connections[conversation_id]):
                try:
                    await connection.send_text(message)
                except Exception:
                    # Remove stale/broken connections
                    self.disconnect(connection, conversation_id)


manager = ConnectionManager()


@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Send a chat message and get AI response.
    
    - **message**: User message content
    - **conversation_id**: Optional existing conversation ID - still buggy, remove this in your request for now
    - **session_id**: Optional session ID for new conversations - still buggy, remove this in your request for now
    - **use_rag**: Whether to use RAG for context (default: True)
    - **max_tokens**: Maximum tokens for response
    - **temperature**: Response creativity (0.0-1.0)
    """
    try:
        # Get or create conversation
        conversation = None
        if chat_request.conversation_id:
            conversation = db.query(Conversation)\
                .filter(Conversation.id == chat_request.conversation_id)\
                .first()
            if not conversation:
                print("error")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
        else:
            # Create new conversation
            session_id = chat_request.session_id or str(uuid.uuid4())
            conversation_data = ConversationCreate(session_id=session_id)
            conversation = Conversation(**conversation_data.dict())
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        # Save user message
        user_message_data = MessageCreate(
            conversation_id=conversation.id,
            content=chat_request.message,
            role=MessageRole.USER,
            message_type=MessageType.TEXT
        )
        user_message = Message(**user_message_data.dict())
        db.add(user_message)
        db.commit()
        
        # Generate AI response using RAG
        response_data = await rag_service.generate_rag_response(
            user_query=chat_request.message,
            conversation_id=conversation.id,
            db=db,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        # Save AI message
        ai_message_data = MessageCreate(
            conversation_id=conversation.id,
            content=response_data['response'],
            role=MessageRole.ASSISTANT,
            message_type=MessageType.TEXT
        )
        ai_message = Message(**ai_message_data.dict())
        
        # Add RAG metadata
        ai_message.sources_used = json.dumps(response_data.get('sources_used', []))
        ai_message.context_chunks = json.dumps(response_data.get('context_chunks', []))
        ai_message.relevance_score = response_data.get('relevance_score')
        ai_message.tokens_used = response_data.get('tokens_used')
        ai_message.processing_time = response_data.get('processing_time')
        ai_message.model_used = response_data.get('model_used')
        
        db.add(ai_message)
        
        # Update conversation metadata
        conversation.total_messages += 2  # user + assistant
        conversation.last_message_at = datetime.utcnow()
        
        # Generate title for new conversations
        if conversation.total_messages == 2 and not conversation.title:
            title = await rag_service.llm_service.generate_conversation_title(
                chat_request.message
            )
            conversation.title = title
        
        db.commit()
        db.refresh(ai_message)
        db.refresh(conversation)
        
        logger.info(f"Chat message processed for conversation {conversation.id}")
        
        return ChatResponse(
            message=MessageResponse.from_orm(ai_message),
            conversation=ConversationResponse.from_orm(conversation),
            processing_info={
                'tokens_used': response_data.get('tokens_used', 0),
                'processing_time': response_data.get('processing_time', '0ms'),
                'sources_count': len(response_data.get('sources_used', [])),
                'relevance_score': response_data.get('relevance_score', 'None')
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = 1,
    size: int = 20,
    db: Session = Depends(get_db)
):
    """Get paginated list of conversations."""
    try:
        # Get total count
        total = db.query(Conversation).count()
        
        # Apply pagination
        offset = (page - 1) * size
        conversations = db.query(Conversation)\
            .filter(Conversation.is_active == True)\
            .order_by(Conversation.last_message_at.desc())\
            .offset(offset)\
            .limit(size)\
            .all()
        
        # Calculate pagination info
        has_next = offset + size < total
        has_prev = page > 1
        
        return ConversationListResponse(
            conversations=[ConversationResponse.from_orm(conv) for conv in conversations],
            total=total,
            page=page,
            size=size,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/history", response_model=ChatHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get full conversation history."""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message)\
        .filter(Message.conversation_id == conversation_id)\
        .order_by(Message.created_at.asc())\
        .all()
    
    return ChatHistoryResponse(
        conversation=ConversationResponse.from_orm(conversation),
        messages=[MessageResponse.from_orm(msg) for msg in messages],
        total_messages=len(messages)
    )


@router.put("/conversations/{conversation_id}")
async def update_conversation_title(
    conversation_id: int,
    title_data: dict,
    db: Session = Depends(get_db)
):
    """Update conversation title."""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    try:
        # Update title
        new_title = title_data.get("title", "").strip()
        if not new_title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title cannot be empty"
            )
        
        conversation.title = new_title
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Conversation title updated: {conversation_id} -> '{new_title}'")
        return ConversationResponse.from_orm(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation title {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating conversation title: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    conversation = db.query(Conversation)\
        .filter(Conversation.id == conversation_id)\
        .first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    try:
        # Delete all messages (cascade should handle this, but explicit is better)
        db.query(Message)\
            .filter(Message.conversation_id == conversation_id)\
            .delete()
        
        # Delete conversation
        db.delete(conversation)
        db.commit()
        
        logger.info(f"Conversation deleted: {conversation_id}")
        return {"message": "Conversation deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}"
        )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time chat.
    
    Expected message format:
    {
        "type": "chat_message",
        "message": "User message content",
        "conversation_id": 123,  // optional
        "use_rag": true,         // optional
        "max_tokens": 1000,      // optional
        "temperature": 0.7       // optional
    }
    """
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "chat_message":
                    # Create chat request
                    chat_request = ChatRequest(
                        message=message_data["message"],
                        conversation_id=message_data.get("conversation_id"),
                        session_id=session_id,
                        use_rag=message_data.get("use_rag", True),
                        max_tokens=message_data.get("max_tokens"),
                        temperature=message_data.get("temperature")
                    )
                    
                    # Send typing indicator
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "typing",
                            "status": "started"
                        }),
                        websocket
                    )
                    
                    # Process message (reuse the send_message logic)
                    try:
                        # Get or create conversation
                        conversation = None
                        if chat_request.conversation_id:
                            conversation = db.query(Conversation)\
                                .filter(Conversation.id == chat_request.conversation_id)\
                                .first()
                        
                        if not conversation:
                            # Try to find existing conversation by session_id only if no conversation_id was provided
                            if not chat_request.conversation_id:
                                conversation = db.query(Conversation)\
                                    .filter(Conversation.session_id == session_id)\
                                    .first()
                        
                        if not conversation:
                            # Create new conversation only if none exists
                            conversation_data = ConversationCreate(session_id=session_id)
                            conversation = Conversation(**conversation_data.dict())
                            db.add(conversation)
                            db.commit()
                            db.refresh(conversation)
                        
                        # Save user message
                        user_message_data = MessageCreate(
                            conversation_id=conversation.id,
                            content=chat_request.message,
                            role=MessageRole.USER,
                            message_type=MessageType.TEXT
                        )
                        user_message = Message(**user_message_data.dict())
                        db.add(user_message)
                        db.commit()
                        
                        # Generate AI response
                        response_data = await rag_service.generate_rag_response(
                            user_query=chat_request.message,
                            conversation_id=conversation.id,
                            db=db,
                            max_tokens=chat_request.max_tokens,
                            temperature=chat_request.temperature
                        )
                        #response_data['response'] = response_data['context_info']['context_chunks'][0]
                        # Save AI message
                        ai_message_data = MessageCreate(
                            conversation_id=conversation.id,
                            content=response_data['response'],
                            role=MessageRole.ASSISTANT,
                            message_type=MessageType.TEXT
                        )
                        ai_message = Message(**ai_message_data.dict())
                        
                        # Add RAG metadata
                        ai_message.sources_used = json.dumps(response_data.get('sources_used', []))
                        ai_message.context_chunks = json.dumps(response_data.get('context_chunks', []))
                        ai_message.relevance_score = response_data.get('relevance_score')
                        ai_message.tokens_used = response_data.get('tokens_used')
                        ai_message.processing_time = response_data.get('processing_time')
                        ai_message.model_used = response_data.get('model_used')
                        
                        db.add(ai_message)
                        
                        # Update conversation
                        conversation.total_messages += 2
                        conversation.last_message_at = datetime.utcnow()
                        
                        if conversation.total_messages == 2 and not conversation.title:
                            title = await rag_service.llm_service.generate_conversation_title(
                                chat_request.message
                            )
                            conversation.title = title
                        
                        db.commit()
                        db.refresh(ai_message)
                        db.refresh(conversation)
                        
                        logger.info(f"AI message after refresh: id={ai_message.id}, content='{ai_message.content[:50]}...', role={ai_message.role}")
                        logger.info(f"Response data: {response_data}")
                        
                        # Send response
                        response_message = {
                            "type": "chat_response",
                            "message": {
                                "id": ai_message.id,
                                "content": ai_message.content,
                                "role": ai_message.role,
                                "created_at": ai_message.created_at.isoformat(),
                                "sources_used": response_data.get('sources_used', []),
                                "relevance_score": response_data.get('relevance_score')
                            },
                            "conversation": {
                                "id": conversation.id,
                                "session_id": conversation.session_id,
                                "title": conversation.title,
                                "total_messages": conversation.total_messages
                            },
                            "processing_info": {
                                "tokens_used": response_data.get('tokens_used', 0),
                                "processing_time": response_data.get('processing_time', '0ms'),
                                "sources_count": len(response_data.get('sources_used', []))
                            }
                        }
                        
                        logger.info(f"Sending WebSocket response: {json.dumps(response_message, indent=2, default=str)}")
                        
                        await manager.send_personal_message(
                            json.dumps(response_message),
                            websocket
                        )
                        
                    except Exception as e:
                        # Send error message
                        error_message = {
                            "type": "error",
                            "message": f"Error processing message: {str(e)}"
                        }
                        await manager.send_personal_message(
                            json.dumps(error_message),
                            websocket
                        )
                    
                    finally:
                        # Stop typing indicator
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "typing",
                                "status": "stopped"
                            }),
                            websocket
                        )
                
                else:
                    # Unknown message type
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "Unknown message type"
                        }),
                        websocket
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }),
                    websocket
                )
                
    except WebSocketDisconnect as e:
        manager.disconnect(websocket, session_id)
        try:
            code = getattr(e, "code", None)
        except Exception:
            code = None
        logger.info(f"WebSocket disconnected: {session_id} code={code}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, session_id)
