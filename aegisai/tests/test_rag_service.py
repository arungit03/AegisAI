"""Tests for the RAG service pipeline."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

from app.rag.service import RAGService
from app.rag.types import RAGQuery, RAGResult
from app.rag.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self):
        self._dimension = 768

    @property
    def dimension(self):
        return self._dimension

    def embed(self, text: str):
        # Return a deterministic mock embedding
        return [0.1] * 768

    def embed_batch(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_rag_service_initialization():
    """Test RAG service can be initialized."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
                top_k=5,
                chunk_size=512,
                chunk_overlap=50,
            )

            assert service is not None
            assert service.top_k == 5
            assert service.chunk_size == 512
            assert service.chunk_overlap == 50
            assert service.llm_model_name == "qwen2.5:7b-instruct"


def test_rag_service_query_with_empty_results():
    """Test RAG query when no results are found."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            # Mock search to return empty
            qdrant.search = MagicMock(return_value=[])

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
                top_k=5,
            )

            query = RAGQuery(
                question="What is the meaning of life?",
                user_role="employee",
                user_id=None,
                department=None,
            )

            with patch.object(service, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "response": "I couldn't find sufficient information...",
                    "tokens_used": 50,
                }

                result = service.query(query)

                assert isinstance(result, RAGResult)
                assert result.question == "What is the meaning of life?"
                assert result.model_used == "qwen2.5:7b-instruct"
                assert result.sources == []


def test_rag_service_query_with_results():
    """Test RAG query with search results."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            # Mock search to return results
            qdrant.search = MagicMock(return_value=[
                {
                    "id": "chunk_1",
                    "score": 0.95,
                    "payload": {
                        "document_id": "doc-123",
                        "filename": "policy.txt",
                        "chunk_text": "The company policy states that remote work is allowed.",
                        "page_number": 1,
                        "classification": "public_internal",
                        "department": "engineering",
                    }
                },
                {
                    "id": "chunk_2",
                    "score": 0.88,
                    "payload": {
                        "document_id": "doc-456",
                        "filename": "guide.md",
                        "chunk_text": "Employees can work remotely up to 3 days per week.",
                        "page_number": 2,
                        "classification": "public_internal",
                        "department": "engineering",
                    }
                }
            ])

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
                top_k=5,
            )

            query = RAGQuery(
                question="Can I work from home?",
                user_role="employee",
                user_id=None,
                department="engineering",
            )

            with patch.object(service, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "response": "Yes, remote work is allowed up to 3 days per week.",
                    "tokens_used": 50,
                }

                result = service.query(query)

                assert isinstance(result, RAGResult)
                assert len(result.sources) == 2
                assert result.sources[0].filename == "policy.txt"
                assert result.sources[1].filename == "guide.md"


def test_rag_service_build_prompt():
    """Test prompt construction."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
            )

            prompt = service._build_prompt(
                question="What is the policy?",
                context="[Source: policy.txt]\nRemote work is allowed."
            )

            assert "What is the policy?" in prompt
            assert "Remote work is allowed." in context
            assert "AegisAI" in prompt
            assert "IMPORTANT INSTRUCTIONS" in prompt


def test_rag_service_query_admin_bypass():
    """Test that admin role gets no permission filter."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            # Track what filter is passed to search
            called_filters = []
            original_search = qdrant.search

            def mock_search(*args, **kwargs):
                called_filters.append(kwargs.get('filter_conditions'))
                return []

            qdrant.search = mock_search

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
            )

            query = RAGQuery(
                question="Test question",
                user_role="admin",
                user_id=None,
                department=None,
            )

            with patch.object(service, '_call_llm') as mock_llm:
                mock_llm.return_value = {"response": "test", "tokens_used": 0}
                service.query(query)

                # Admin filter should be empty (no restrictions)
                assert len(called_filters) == 1
                admin_filter = called_filters[0]
                assert admin_filter.must == []


def test_rag_service_query_permission_filter_applied():
    """Test that permission filters are applied at Qdrant search level."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            # Track filter
            called_filters = []

            def mock_search(*args, **kwargs):
                called_filters.append(kwargs.get('filter_conditions'))
                return []

            qdrant.search = mock_search

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
            )

            query = RAGQuery(
                question="Test question",
                user_role="employee",
                user_id=uuid4(),
                department="engineering",
            )

            with patch.object(service, '_call_llm') as mock_llm:
                mock_llm.return_value = {"response": "test", "tokens_used": 0}
                service.query(query)

                # Employee filter should have should conditions
                assert len(called_filters) == 1
                employee_filter = called_filters[0]
                assert len(employee_filter.should) > 0


def test_rag_service_query_exception_handling():
    """Test that RAG service handles exceptions gracefully."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(result=[])

        with patch('app.rag.service.get_embedding_model') as mock_get_model:
            mock_model = MockEmbeddingProvider()
            mock_get_model.return_value = mock_model

            from app.rag.qdrant import QdrantManager
            qdrant = QdrantManager(url="http://localhost:6333", api_key=None)

            # Mock search to raise exception
            qdrant.search = MagicMock(side_effect=Exception("Connection failed"))

            service = RAGService(
                qdrant_manager=qdrant,
                embedding_model_name="nomic-embed-text",
                llm_model_name="qwen2.5:7b-instruct",
            )

            query = RAGQuery(
                question="Test question",
                user_role="employee",
                user_id=None,
                department=None,
            )

            # Should not raise - should return error response
            result = service.query(query)
            assert isinstance(result, RAGResult)
            assert "error" in result.answer.lower() or "sorry" in result.answer.lower()
            assert result.sources == []
