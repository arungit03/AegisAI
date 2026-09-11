"""Conversation Memory Service for PostgreSQL-persisted context window management.

This service handles:
1. Storing conversation history in PostgreSQL
2. Context window management with automatic summarization
3. Retrieving relevant context for RAG queries
4. Memory compaction when token limits are exceeded
"""

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.chat import Conversation, Message, ConversationMemory, MessageRole
from app.rag.service import RAGService


class ConversationMemoryService:
    """Service for managing conversation memory with PostgreSQL persistence."""

    def __init__(self, db: AsyncSession):
        self.db = db
        # Context window settings
        self.max_context_messages = getattr(settings, 'MAX_CONTEXT_MESSAGES', 20)
        self.max_context_tokens = getattr(settings, 'MAX_CONTEXT_TOKENS', 4096)
        self.summarize_threshold = getattr(settings, 'SUMMARIZE_THRESHOLD', 15)

    async def get_conversation_context(
        self,
        conversation_id: UUID,
        include_memory: bool = True
    ) -> Tuple[List[Message], Optional[str]]:
        """Get conversation context for RAG queries.

        Returns:
            Tuple of (recent_messages, memory_summary)
            - recent_messages: Most recent messages within token limit
            - memory_summary: Summarized context from older messages
        """
        # Get recent messages
        recent_messages = await self._get_recent_messages(conversation_id)

        # Get memory summary if requested
        memory_summary = None
        if include_memory:
            memory_summary = await self._get_memory_summary(conversation_id)

        return recent_messages, memory_summary

    async def _get_recent_messages(
        self,
        conversation_id: UUID,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get recent messages within token budget."""
        if limit is None:
            limit = self.max_context_messages

        query = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        messages = list(reversed(result.scalars().all()))

        # Apply token budget
        return self._apply_token_budget(messages)

    def _apply_token_budget(self, messages: List[Message]) -> List[Message]:
        """Apply token budget to message list, keeping most recent."""
        if not messages:
            return []

        total_tokens = 0
        result = []

        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        for msg in reversed(messages):
            msg_tokens = msg.token_count or max(1, len(msg.content) // 4)
            if total_tokens + msg_tokens > self.max_context_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens

        return result

    async def _get_memory_summary(self, conversation_id: UUID) -> Optional[str]:
        """Get the latest memory summary for a conversation."""
        query = select(ConversationMemory).where(
            ConversationMemory.conversation_id == conversation_id
        ).order_by(ConversationMemory.window_end.desc()).limit(1)

        result = await self.db.execute(query)
        memory = result.scalar_one_or_none()

        return memory.summary if memory else None

    async def add_message_and_manage_memory(
        self,
        message: Message,
        rag_service: Optional[RAGService] = None
    ) -> None:
        """Add a message and manage memory compaction if needed."""
        conversation_id = message.conversation_id

        # Check if we need to summarize older messages
        message_count = await self._get_message_count(conversation_id)

        if message_count >= self.summarize_threshold:
            await self._compact_memory(conversation_id, rag_service)

    async def _get_message_count(self, conversation_id: UUID) -> int:
        """Get total message count for a conversation."""
        query = select(func.count(Message.id)).where(
            Message.conversation_id == conversation_id
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def _compact_memory(
        self,
        conversation_id: UUID,
        rag_service: Optional[RAGService] = None
    ) -> None:
        """Compact older messages into a memory summary."""
        # Get messages that are not yet summarized
        # Find the latest memory window_end
        latest_memory = await self._get_latest_memory(conversation_id)

        if latest_memory:
            # Get messages after the latest memory window
            query = select(Message).where(
                and_(
                    Message.conversation_id == conversation_id,
                    Message.created_at > latest_memory.window_end
                )
            ).order_by(Message.created_at.asc())
        else:
            # Get all messages except the most recent ones (keep recent in active window)
            query = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.asc()).limit(-self.max_context_messages)

        result = await self.db.execute(query)
        messages_to_summarize = result.scalars().all()

        if len(messages_to_summarize) < 2:
            return  # Not enough messages to summarize

        # Generate summary using LLM or simple extraction
        summary_text = await self._generate_summary(messages_to_summarize, rag_service)

        # Create memory record
        memory = ConversationMemory(
            conversation_id=conversation_id,
            summary=summary_text,
            message_count=len(messages_to_summarize),
            token_count=sum(m.token_count or max(1, len(m.content) // 4) for m in messages_to_summarize),
            window_start=messages_to_summarize[0].created_at,
            window_end=messages_to_summarize[-1].created_at,
        )

        self.db.add(memory)
        await self.db.commit()

    async def _get_latest_memory(self, conversation_id: UUID) -> Optional[ConversationMemory]:
        """Get the latest memory record for a conversation."""
        query = select(ConversationMemory).where(
            ConversationMemory.conversation_id == conversation_id
        ).order_by(ConversationMemory.window_end.desc()).limit(1)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _generate_summary(
        self,
        messages: List[Message],
        rag_service: Optional[RAGService] = None
    ) -> str:
        """Generate a summary of messages.

        If rag_service is provided, uses LLM for summarization.
        Otherwise, uses simple extraction.
        """
        if rag_service:
            return await self._llm_summarize(messages, rag_service)
        else:
            return self._simple_summarize(messages)

    def _simple_summarize(self, messages: List[Message]) -> str:
        """Simple extractive summarization."""
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]

        summary_parts = []

        if user_messages:
            topics = [m.content[:100] for m in user_messages[:5]]
            summary_parts.append("User asked about: " + "; ".join(topics))

        if assistant_messages:
            # Include key information from assistant responses
            key_info = []
            for m in assistant_messages[:3]:
                # Extract first sentence or key points
                first_sentence = m.content.split('.')[0][:150]
                key_info.append(first_sentence)
            if key_info:
                summary_parts.append("Key responses: " + "; ".join(key_info))

        return " | ".join(summary_parts) if summary_parts else f"Summary of {len(messages)} messages"

    async def _llm_summarize(
        self,
        messages: List[Message],
        rag_service: RAGService
    ) -> str:
        """Generate summary using LLM."""
        conversation_text = "\n".join([
            f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {m.content}"
            for m in messages
        ])

        prompt = f"""Summarize the following conversation in 3-4 sentences, focusing on key topics, decisions, and information exchanged:

{conversation_text}

Summary:"""

        try:
            summary = rag_service._call_llm(prompt, temperature=0.3, max_tokens=256)
            return summary.strip()
        except Exception:
            # Fallback to simple summarization
            return self._simple_summarize(messages)

    async def get_full_conversation_history(
        self,
        conversation_id: UUID,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[Message], int]:
        """Get full conversation history with pagination."""
        # Total count
        count_query = select(func.count()).select_from(
            select(Message).where(Message.conversation_id == conversation_id).subquery()
        )
        total = await self.db.scalar(count_query)

        # Paginated messages
        query = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.asc()).offset(
            (page - 1) * page_size
        ).limit(page_size)

        result = await self.db.execute(query)
        messages = result.scalars().all()

        return list(messages), total

    async def get_memory_timeline(self, conversation_id: UUID) -> List[ConversationMemory]:
        """Get all memory summaries for a conversation in chronological order."""
        query = select(ConversationMemory).where(
            ConversationMemory.conversation_id == conversation_id
        ).order_by(ConversationMemory.window_start.asc())

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def clear_memory(self, conversation_id: UUID) -> None:
        """Clear all memory for a conversation (e.g., on user request)."""
        query = select(ConversationMemory).where(
            ConversationMemory.conversation_id == conversation_id
        )
        result = await self.db.execute(query)
        memories = result.scalars().all()

        for memory in memories:
            await self.db.delete(memory)

        await self.db.commit()

    async def rebuild_memory(
        self,
        conversation_id: UUID,
        rag_service: Optional[RAGService] = None
    ) -> None:
        """Rebuild memory from scratch (useful after settings change)."""
        # Clear existing memory
        await self.clear_memory(conversation_id)

        # Get all messages
        all_messages, _ = await self.get_full_conversation_history(conversation_id, page_size=10000)

        if len(all_messages) <= self.max_context_messages:
            return  # Not enough messages to need memory

        # Split into chunks and summarize each
        chunk_size = self.max_context_messages
        for i in range(0, len(all_messages) - self.max_context_messages, chunk_size):
            chunk = all_messages[i:i + chunk_size]
            if len(chunk) < 2:
                continue

            summary_text = await self._generate_summary(chunk, rag_service)

            memory = ConversationMemory(
                conversation_id=conversation_id,
                summary=summary_text,
                message_count=len(chunk),
                token_count=sum(m.token_count or max(1, len(m.content) // 4) for m in chunk),
                window_start=chunk[0].created_at,
                window_end=chunk[-1].created_at,
            )
            self.db.add(memory)

        await self.db.commit()