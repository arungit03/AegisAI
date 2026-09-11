# Implementation Plan: Streaming Chat, High-Quality Citations, Conversation Memory

## Overview
This plan details the implementation of three major features while preserving all existing SIH RAG accuracy fixes (141 passed tests).

---

## 1. Streaming Chat (SSE) - `/api/chat/stream`

### Backend Changes
**File: `aegisai/backend/app/api/v1/chat.py`**
- Add new endpoint `POST /chat/stream` with `stream: bool = True`
- Use `StreamingResponse` with `text/event-stream` media type
- Stream tokens as SSE events: `data: {"token": "...", "done": false}\n\n`
- Final event: `data: {"done": true, "sources": [...], "conversation_id": "..."}\n\n`
- Reuse existing RAG pipeline but call Ollama with `stream: true`

**File: `aegisai/backend/app/rag/service.py`**
- Modify `_call_llm` to support streaming mode
- Add `_call_llm_stream` method that yields tokens from Ollama's streaming API
- Ollama streaming endpoint: `/api/generate` with `"stream": true`
- Parse SSE response from Ollama and yield individual tokens

**File: `aegisai/backend/app/schemas/chat.py`**
- Add `ChatStreamRequest` (extends ChatRequest with stream: Literal[True])
- Add `ChatStreamChunk` for SSE event format
- Keep existing `ChatResponse` for non-streaming

### Frontend Changes
**File: `aegisai/frontend/src/services/api.ts`**
- Add `sendMessageStream` method that returns AsyncIterable/ReadableStream
- Handle SSE parsing on client side

**File: `aegisai/frontend/src/stores/chatStore.ts`**
- Add streaming state management
- Add `addStreamingMessage`, `updateStreamingMessage`, `completeStreamingMessage`
- Handle token accumulation and UI updates

**File: `aegisai/frontend/src/pages/ChatPage.tsx`**
- Use streaming when `stream: true` (default)
- Show tokens as they arrive with typing indicator
- Handle sources when stream completes

---

## 2. High-Quality Citations

### Backend Changes
**File: `aegisai/backend/app/schemas/chat.py`**
- Enhance `SourceCitation` with:
  - `section_title: Optional[str]` - section/heading from document
  - `chunk_index: Optional[int]` - position in document
  - `char_start: Optional[int]`, `char_end: Optional[int]` - character offsets
  - `document_title: Optional[str]` - human-readable title
  - `author: Optional[str]` - document author if available

**File: `aegisai/backend/app/rag/types.py`**
- Enhance `SearchHit` / `RAGSearchHit` with additional metadata fields
- Ensure Qdrant payload includes all citation metadata

**File: `aegisai/backend/app/rag/qdrant.py`**
- Update payload storage to include section titles, chunk positions
- Modify search to return all metadata fields

**File: `aegisai/backend/app/rag/ingestion.py` / `chunking.py`**
- Extract section headings during chunking
- Store section context with each chunk
- Track character offsets

**File: `aegisai/backend/app/api/v1/chat.py`**
- Map enhanced SearchHit fields to SourceCitation in response

### Frontend Changes
**File: `aegisai/frontend/src/types/index.ts`**
- Update `SourceCitation` interface with new fields

**File: `aegisai/frontend/src/pages/ChatPage.tsx`**
- Enhanced citation display with section titles, page numbers
- Hover tooltip with chunk preview
- Click to navigate to document viewer (future)

---

## 3. Conversation Memory

### Backend Changes
**File: `aegisai/backend/app/models/chat.py`**
- Add `ConversationSummary` model for summarized context
- Add `summary` field to `Conversation` model
- Add `turn_count` for context window management

**File: `aegisai/backend/app/api/v1/chat.py`**
- Modify `chat` and `chat/stream` endpoints to include full conversation history
- Add context window management (last N turns or token budget)
- Pass conversation history to RAG query

**File: `aegisai/backend/app/rag/service.py`**
- Update `RAGQuery` to accept `conversation_history` (already exists)
- Ensure memory queries (QueryIntent.MEMORY) use full history
- Implement summarization for long conversations (optional enhancement)

**File: `aegisai/backend/app/schemas/chat.py`**
- Add `ConversationDetailResponse` with full message history
- Add pagination for message loading

### Frontend Changes
**File: `aegisai/frontend/src/stores/chatStore.ts`**
- Add conversation history persistence
- Add "Load More Messages" pagination
- Add conversation switching without losing state

**File: `aegisai/frontend/src/pages/ChatPage.tsx`**
- Show conversation history in sidebar
- Implement infinite scroll for messages
- Preserve scroll position when switching conversations

---

## Integration Points & Testing

### Preserve Existing Behavior
- All 141 existing tests must pass
- SIH structured lookup must remain exact
- Permission filtering must remain at Qdrant level
- Anti-hallucination prompts must remain unchanged

### Test Coverage
- Add tests for SSE streaming endpoint
- Add tests for enhanced citations
- Add tests for conversation memory persistence
- Integration tests for all three features together

### Docker/Deployment
- Update docker-compose if needed (no changes expected)
- Verify Ollama streaming works in container
- Ensure PostgreSQL handles conversation history load

---

## Implementation Order

1. **Backend Streaming** - Core SSE endpoint + Ollama streaming
2. **Frontend Streaming** - SSE consumption + UI updates
3. **Backend Citations** - Enhanced metadata in payload + schema
4. **Frontend Citations** - Enhanced display
5. **Backend Memory** - Conversation history + context window
6. **Frontend Memory** - History UI + pagination
7. **Integration Testing** - Full flow verification
8. **Test Suite** - Run all 141 tests
9. **Docker Build** - Verify deployment