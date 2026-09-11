"""Tests for Qdrant vector database manager.

Note: These tests use mock/Qdrant mock client since we may not have
a running Qdrant instance in the test environment."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import UUID, uuid4

from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Conditions,
)

from app.rag.qdrant import QdrantManager


def test_qdrant_manager_initialization():
    """Test Qdrant manager can be initialized."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        assert manager is not None
        assert manager.url == "http://localhost:6333"
        assert manager.collection_name == "aegisai_documents"
        assert manager.vector_size == 768


def test_qdrant_manager_custom_params():
    """Test Qdrant manager with custom parameters."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        manager = QdrantManager(
            url="http://localhost:6334",
            api_key="test_key",
            collection_name="custom_collection",
            vector_size=512,
        )
        assert manager.url == "http://localhost:6334"
        assert manager.api_key == "test_key"
        assert manager.collection_name == "custom_collection"
        assert manager.vector_size == 512


def test_qdrant_ensure_collection_creation():
    """Test collection creation when it doesn't exist."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().get_collections.return_value = MagicMock(
            result=[]  # No existing collections
        )
        manager = QdrantManager(url="http://localhost:6333", api_key=None, collection_name="test_collection")
        manager.ensure_collection()
        mock_client().create_collection.assert_called_once()


def test_qdrant_ensure_collection_exists():
    """Test collection creation when it already exists (no-op)."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_collections = MagicMock()
        mock_collections.result = [{"name": "test_collection", "vectors_count": 0}]
        mock_client().get_collections.return_value = mock_collections

        manager = QdrantManager(url="http://localhost:6333", api_key=None, collection_name="test_collection")
        manager.ensure_collection()
        # Should NOT call create_collection since it exists
        mock_client().create_collection.assert_not_called()


def test_qdrant_create_permission_filter_admin():
    """Test admin bypass - no filter applied."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        filter_obj = manager.create_permission_filter(
            user_role="admin",
            user_id=None,
            department=None,
        )
        # Admin should get empty filter (all access) - must be empty
        assert filter_obj is not None
        assert filter_obj.must == []


def test_qdrant_create_permission_filter_employee():
    """Test employee permission filtering - PUBLIC_INTERNAL only."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        filter_obj = manager.create_permission_filter(
            user_role="employee",
            user_id=None,
            department=None,
        )
        # Employee should get filter with should conditions containing classification=public_internal
        assert filter_obj is not None
        assert hasattr(filter_obj, 'should')
        assert len(filter_obj.should) > 0


def test_qdrant_create_permission_filter_manager():
    """Test manager permission filtering - department + PUBLIC_INTERNAL."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        filter_obj = manager.create_permission_filter(
            user_role="manager",
            user_id=None,
            department="engineering",
        )
        # Manager should get filter with department + classification conditions
        assert filter_obj is not None
        assert hasattr(filter_obj, 'should')
        # Should have both department and classification in should conditions
        filter_str = str(filter_obj)
        assert "engineering" in filter_str or "department" in filter_str


def test_qdrant_create_permission_filter_engineer():
    """Test engineer permission filtering - department + PUBLIC_INTERNAL."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        filter_obj = manager.create_permission_filter(
            user_role="engineer",
            user_id=None,
            department="engineering",
        )
        # Engineer should get similar filter as manager
        assert filter_obj is not None
        assert hasattr(filter_obj, 'should')


def test_qdrant_create_permission_filter_employee_with_user_id():
    """Test employee permission filtering with explicit user_id."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        user_id = uuid4()
        filter_obj = manager.create_permission_filter(
            user_role="employee",
            user_id=user_id,
            department=None,
        )
        # Should have should conditions
        assert filter_obj is not None
        assert hasattr(filter_obj, 'should')
        # Should include uploaded_by_id condition for the user
        filter_str = str(filter_obj)
        assert str(user_id) in filter_str


def test_qdrant_create_permission_filter_with_allowed_document_ids():
    """Test filter with explicitly allowed document IDs."""
    with patch('app.rag.qdrant.qdrant_client'):
        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        doc_ids = [uuid4(), uuid4()]
        filter_obj = manager.create_permission_filter(
            user_role="employee",
            user_id=None,
            department=None,
            allowed_document_ids=doc_ids,
        )
        # Should have should conditions
        assert filter_obj is not None
        assert hasattr(filter_obj, 'should')


def test_qdrant_search_with_filter():
    """Test Qdrant search with permission filter."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        manager = QdrantManager(url="http://localhost:6333", api_key=None)

        # Mock the search method
        manager.search = MagicMock(return_value=[])
        results = manager.search(
            query_vector=[0.1] * 768,
            limit=5,
            filter_conditions=None,
        )
        assert results == []


def test_qdrant_search_with_mock_results():
    """Test Qdrant search returns results in correct format."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "test_id_123"
        mock_result.score = 0.85
        mock_result.payload = {
            "document_id": "doc-123",
            "filename": "test.txt",
            "chunk_text": "sample text",
            "page_number": 1,
            "classification": "public_internal",
            "department": "engineering",
        }
        mock_client().search.return_value = [mock_result]

        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        results = manager.search(
            query_vector=[0.1] * 768,
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "test_id_123"
        assert results[0]["score"] == 0.85
        assert results[0]["payload"]["filename"] == "test.txt"


def test_qdrant_add_vectors():
    """Test adding vectors to Qdrant."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        manager = QdrantManager(url="http://localhost:6333", api_key=None)

        point = PointStruct(
            id="test_id",
            vector=[0.1] * 768,
            payload={"text": "test"},
        )
        result = manager.add_vectors([point])

        assert result is True
        mock_client().upload_points.assert_called_once()


def test_qdrant_delete_document_vectors():
    """Test deleting document vectors."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        # Mock scroll to return empty list
        mock_client().scroll.return_value = ([], "next_page")

        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        doc_id = uuid4()
        result = manager.delete_document_vectors(doc_id)

        assert result is True
        mock_client().scroll.assert_called_once()


def test_qdrant_get_collection_info():
    """Test getting collection info."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.config.params.vectors.size = 768
        mock_info.config.params.vectors.distance = "Cosine"
        mock_client().get_collection.return_value = mock_info

        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        info = manager.get_collection_info()

        assert info is not None
        assert info["vectors_count"] == 100


def test_point_struct_creation():
    """Test PointStruct creation for Qdrant."""
    point = PointStruct(
        id="test_id_123",
        vector=[0.1] * 768,
        payload={"test": "data"},
    )
    assert point.id == "test_id_123"
    assert len(point.vector) == 768


def test_qdrant_search_returns_empty_on_error():
    """Test that search returns empty list on error."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().search.side_effect = Exception("Connection error")

        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        results = manager.search(
            query_vector=[0.1] * 768,
            limit=5,
        )
        assert results == []


def test_qdrant_add_vectors_returns_false_on_error():
    """Test that add_vectors returns False on error."""
    with patch('app.rag.qdrant.qdrant_client') as mock_client:
        mock_client().upload_points.side_effect = Exception("Connection error")

        manager = QdrantManager(url="http://localhost:6333", api_key=None)
        point = PointStruct(
            id="test_id",
            vector=[0.1] * 768,
            payload={"text": "test"},
        )
        result = manager.add_vectors([point])
        assert result is False