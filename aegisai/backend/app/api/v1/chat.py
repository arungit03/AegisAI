"""Chat/Conversation endpoints."""
from datetime import datetime
from typing import List, Optional
import time
import json

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from uuid import UUID

from app.core.config import settings
from app.core.database import get_db
from app.models.chat import Conversation, Message, MessageRole
from app.models.user import User
from app.schemas.chat import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListResponse,
    MessageResponse,
    ChatRequest,
    ChatResponse,
    SourceCitation,
)
from app.services.audit import audit_log
from app.services.document_awareness import answer_document_awareness, get_authorized_documents
from app.services.conversation_memory import ConversationMemoryService
from app.api.v1.auth import get_current_user
from app.rag.service import RAGService, RAGQuery, RAGResult
from app.rag.qdrant import QdrantManager
from app.rag.routing import QueryIntent, classify_query

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: Request,
    conversation_data: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Create a new conversation."""
    conversation = Conversation(
        title=conversation_data.title,
        user_id=current_user.id,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    await audit_log(db, request, "conversation_created", user_id=current_user.id,
                    resource_type="conversation", resource_id=conversation.id)

    return conversation


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationListResponse:
    """List user's conversations."""
    query = select(Conversation).where(Conversation.user_id == current_user.id)

    if not include_archived:
        query = query.where(Conversation.is_archived == False)

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Pagination
    query = query.order_by(Conversation.updated_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    conversations = result.scalars().all()

    return ConversationListResponse(conversations=conversations, total=total, page=page, page_size=page_size)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Get a specific conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    request: Request,
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Update a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    update_data = conversation_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(conversation, field, value)

    await db.commit()
    await db.refresh(conversation)

    await audit_log(db, request, "conversation_updated", user_id=current_user.id,
                    resource_type="conversation", resource_id=conversation.id)

    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    request: Request,
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()

    await audit_log(db, request, "conversation_deleted", user_id=current_user.id,
                    resource_type="conversation", resource_id=conversation.id)


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[MessageResponse]:
    """Get messages for a conversation."""
    # Verify conversation ownership
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Get messages with pagination
    query = select(Message).where(Message.conversation_id == conversation_id)
    query = query.order_by(Message.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    messages = result.scalars().all()

    # Return in chronological order
    return list(reversed(messages))


@router.get("/conversations/{conversation_id}/memory")
async def get_conversation_memory(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get conversation memory timeline (summaries of older context)."""
    # Verify conversation ownership
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Get memory service
    memory_service = ConversationMemoryService(db)
    memory_timeline = await memory_service.get_memory_timeline(conversation_id)

    return [
        {
            "id": str(memory.id),
            "summary": memory.summary,
            "message_count": memory.message_count,
            "token_count": memory.token_count,
            "window_start": memory.window_start.isoformat(),
            "window_end": memory.window_end.isoformat(),
            "created_at": memory.created_at.isoformat(),
        }
        for memory in memory_timeline
    ]


@router.post("", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Send a chat message and get AI response with RAG.

    Integrates the local RAG pipeline:
    1. Permission-aware retrieval from Qdrant
    2. Context construction from authorized chunks
    3. Local LLM inference via Ollama
    4. Grounded answer with source citations

    The critical security step: authorization filtering happens at the
    Qdrant search level, ensuring unauthorized chunks are never sent
    to the LLM.
    """
    start_time = time.time()
    conversation_history = []
    memory_summary = None

    # Initialize memory service
    memory_service = ConversationMemoryService(db)

    # Get or create conversation
    if chat_request.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == chat_request.conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

        # Get conversation context with memory
        recent_messages, memory_summary = await memory_service.get_conversation_context(conversation.id)
        conversation_history = [
            {
                "role": getattr(message.role, "value", str(message.role)),
                "content": message.content,
            }
            for message in recent_messages
        ]
    else:
        # Create new conversation with first message as title
        title = chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message
        conversation = Conversation(title=title, user_id=current_user.id)
        db.add(conversation)
        await db.flush()

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=chat_request.message,
    )
    db.add(user_message)
    await db.flush()  # Flush to get the message ID

    intent = classify_query(chat_request.message)

    if intent == QueryIntent.DOCUMENT_AWARE:
        owner_only = bool(
            any(term in chat_request.message.lower() for term in ("my", " i ", "added", "uploaded"))
        )
        documents = await get_authorized_documents(db, current_user, owner_only=owner_only)
        matching_answer = answer_document_awareness(chat_request.message, documents)
        rag_result = RAGResult(
            answer=matching_answer,
            sources=[],
            model_used="document-awareness",
            processing_time_ms=int((time.time() - start_time) * 1000),
            tokens_used=0,
            question=chat_request.message,
        )
    else:
        # Only company-specific requests initialize and query Qdrant.
        qdrant = None
        if intent == QueryIntent.COMPANY:
            qdrant = QdrantManager(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vector_size=settings.EMBEDDING_DIMENSION,
            )

        rag_service = RAGService(
            qdrant_manager=qdrant,
            embedding_model_name=settings.OLLAMA_EMBEDDING_MODEL or "nomic-embed-text",
            llm_model_name=settings.OLLAMA_MODEL or "qwen2.5:7b-instruct",
            top_k=settings.TOP_K_RESULTS or 5,
            chunk_size=settings.CHUNK_SIZE or 512,
            chunk_overlap=settings.CHUNK_OVERLAP or 50,
        )

        # Execute the existing RAG pipeline for document-content questions.
        rag_query = RAGQuery(
            question=chat_request.message,
            conversation_id=conversation.id,
            user_role=current_user.role.name if current_user.role else "employee",
            user_id=current_user.id,
            department=current_user.department.name if current_user.department else None,
            intent=intent.value,
            conversation_history=conversation_history,
            memory_summary=memory_summary,
            enable_hybrid=chat_request.enable_hybrid,
        )

        rag_result = rag_service.query(rag_query)

    # Save assistant message with RAG or document-awareness results.
    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=rag_result.answer,
        model_used=rag_result.model_used,
        processing_time_ms=rag_result.processing_time_ms,
        token_count=rag_result.tokens_used,
    )
    db.add(assistant_message)

    # Update conversation message count and timestamp
    conversation.message_count += 2
    conversation.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(assistant_message)

    # Manage conversation memory (compact if needed)
    await memory_service.add_message_and_manage_memory(assistant_message, rag_service if intent != QueryIntent.DOCUMENT_AWARE else None)

    processing_time = int((time.time() - start_time) * 1000)

    # Audit log the RAG query
    await audit_log(db, request, "chat_query", user_id=current_user.id,
                    resource_type="conversation", resource_id=conversation.id,
                    details={
                        "message_length": len(chat_request.message),
                        "sources_count": len(rag_result.sources),
                        "answer_length": len(rag_result.answer),
                        "processing_time_ms": processing_time,
                    })

    # Build source citations
    sources = []
    for source in rag_result.sources:
        sources.append(SourceCitation(
            document_id=source.document_id,
            document_filename=source.filename,
            document_title=source.document_title,
            page_number=source.page_number,
            chunk_text=source.chunk_text,
            similarity_score=source.score,
            section_title=source.section_title,
            chunk_index=source.chunk_index,
            char_start=source.char_start,
            char_end=source.char_end,
            author=source.author,
        ))

    return ChatResponse(
        message=rag_result.answer,
        conversation_id=conversation.id,
        sources=sources,
        processing_time_ms=processing_time,
        token_count=rag_result.tokens_used,
        confidence=getattr(rag_result, "confidence", None),
        pipeline=getattr(rag_result, "pipeline", "legacy"),
    )


@router.post("/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a chat message and get AI response with RAG via Server-Sent Events (SSE).

    This endpoint streams tokens in real-time as they are generated by the LLM.
    Each event is a JSON object with either a 'token' field (for streaming content)
    or a 'done' field with final metadata (sources, conversation_id, etc.).

    Event format:
    - Token events: data: {"token": "partial text", "done": false}\n\n
    - Final event: data: {"done": true, "sources": [...], "conversation_id": "...", "processing_time_ms": ..., "token_count": ...}\n\n
    """
    start_time = time.time()
    conversation_history = []
    memory_summary = None

    # Initialize memory service
    memory_service = ConversationMemoryService(db)

    # Get or create conversation
    if chat_request.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == chat_request.conversation_id,
                Conversation.user_id == current_user.id
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

        # Get conversation context with memory
        recent_messages, memory_summary = await memory_service.get_conversation_context(conversation.id)
        conversation_history = [
            {
                "role": getattr(message.role, "value", str(message.role)),
                "content": message.content,
            }
            for message in recent_messages
        ]
    else:
        # Create new conversation with first message as title
        title = chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message
        conversation = Conversation(title=title, user_id=current_user.id)
        db.add(conversation)
        await db.flush()

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=chat_request.message,
    )
    db.add(user_message)
    await db.flush()  # Flush to get the message ID
    await db.commit()

    intent = classify_query(chat_request.message)

    if intent == QueryIntent.DOCUMENT_AWARE:
        owner_only = bool(
            any(term in chat_request.message.lower() for term in ("my", " i ", "added", "uploaded"))
        )
        documents = await get_authorized_documents(db, current_user, owner_only=owner_only)
        matching_answer = answer_document_awareness(chat_request.message, documents)
        rag_result = RAGResult(
            answer=matching_answer,
            sources=[],
            model_used="document-awareness",
            processing_time_ms=int((time.time() - start_time) * 1000),
            tokens_used=0,
            question=chat_request.message,
        )

        # For document-aware, stream the pre-computed answer
        async def generate_document_aware():
            # Stream the answer word by word for effect
            words = matching_answer.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'token': word + (' ' if i < len(words) - 1 else ''), 'done': False})}\n\n"
            # Final event with metadata
            processing_time = int((time.time() - start_time) * 1000)
            sources_json = []
            yield f"data: {json.dumps({'done': True, 'sources': sources_json, 'conversation_id': str(conversation.id), 'processing_time_ms': processing_time, 'token_count': 0})}\n\n"

        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=rag_result.answer,
            model_used=rag_result.model_used,
            processing_time_ms=rag_result.processing_time_ms,
            token_count=rag_result.tokens_used,
        )
        db.add(assistant_message)
        conversation.message_count += 2
        conversation.updated_at = datetime.utcnow()
        await db.commit()

        # Manage conversation memory (compact if needed)
        await memory_service.add_message_and_manage_memory(assistant_message, None)

        await audit_log(db, request, "chat_query", user_id=current_user.id,
                        resource_type="conversation", resource_id=conversation.id,
                        details={
                            "message_length": len(chat_request.message),
                            "sources_count": len(rag_result.sources),
                            "answer_length": len(rag_result.answer),
                            "processing_time_ms": processing_time,
                        })

        return StreamingResponse(
            generate_document_aware(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    else:
        # Only company-specific requests initialize and query Qdrant.
        qdrant = None
        if intent == QueryIntent.COMPANY:
            qdrant = QdrantManager(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vector_size=settings.EMBEDDING_DIMENSION,
            )

        rag_service = RAGService(
            qdrant_manager=qdrant,
            embedding_model_name=settings.OLLAMA_EMBEDDING_MODEL or "nomic-embed-text",
            llm_model_name=settings.OLLAMA_MODEL or "qwen2.5:7b-instruct",
            top_k=settings.TOP_K_RESULTS or 5,
            chunk_size=settings.CHUNK_SIZE or 512,
            chunk_overlap=settings.CHUNK_OVERLAP or 50,
        )

        # Execute the existing RAG pipeline for document-content questions.
        rag_query = RAGQuery(
            question=chat_request.message,
            conversation_id=conversation.id,
            user_role=current_user.role.name if current_user.role else "employee",
            user_id=current_user.id,
            department=current_user.department.name if current_user.department else None,
            intent=intent.value,
            conversation_history=conversation_history,
            memory_summary=memory_summary,
            enable_hybrid=chat_request.enable_hybrid,
        )

        # Run RAG retrieval first (non-streaming)
        rag_result = rag_service.query(rag_query)

        # Build source citations
        sources = []
        for source in rag_result.sources:
            sources.append(SourceCitation(
                document_id=source.document_id,
                document_filename=source.filename,
                document_title=source.document_title,
                page_number=source.page_number,
                chunk_text=source.chunk_text,
                similarity_score=source.score,
                section_title=source.section_title,
                chunk_index=source.chunk_index,
                char_start=source.char_start,
                char_end=source.char_end,
                author=source.author,
            ))

        # Create streaming generator
        async def generate_stream():
            # Stream the answer from the LLM
            full_answer = ""
            prompt = rag_service._build_prompt(
                rag_query.question,
                "\n\n".join([f"[Source: {hit.filename}]\n{hit.chunk_text}" for hit in rag_result.sources]),
                rag_query.conversation_history
            )

            # Use streaming LLM call
            for token in rag_service._call_llm_stream(prompt):
                full_answer += token
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

            # Final event with metadata
            processing_time = int((time.time() - start_time) * 1000)
            sources_json = [
                {
                    "document_id": str(s.document_id),
                    "document_filename": s.document_filename,
                    "document_title": s.document_title,
                    "page_number": s.page_number,
                    "chunk_text": s.chunk_text,
                    "similarity_score": s.similarity_score,
                    "section_title": s.section_title,
                    "chunk_index": s.chunk_index,
                    "char_start": s.char_start,
                    "char_end": s.char_end,
                    "author": s.author,
                }
                for s in sources
            ]
            yield f"data: {json.dumps({'done': True, 'sources': sources_json, 'conversation_id': str(conversation.id), 'processing_time_ms': processing_time, 'token_count': rag_result.tokens_used, 'confidence': getattr(rag_result, 'confidence', None), 'pipeline': getattr(rag_result, 'pipeline', 'legacy')})}\n\n"

            # Save assistant message after streaming completes
            assistant_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=full_answer if full_answer else rag_result.answer,
                model_used=rag_result.model_used,
                processing_time_ms=processing_time,
                token_count=rag_result.tokens_used,
            )
            db.add(assistant_message)
            conversation.message_count += 2
            conversation.updated_at = datetime.utcnow()
            await db.commit()

            # Manage conversation memory (compact if needed)
            await memory_service.add_message_and_manage_memory(assistant_message, rag_service)

            await audit_log(db, request, "chat_query", user_id=current_user.id,
                            resource_type="conversation", resource_id=conversation.id,
                            details={
                                "message_length": len(chat_request.message),
                                "sources_count": len(rag_result.sources),
                                "answer_length": len(full_answer if full_answer else rag_result.answer),
                                "processing_time_ms": processing_time,
                            })

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
