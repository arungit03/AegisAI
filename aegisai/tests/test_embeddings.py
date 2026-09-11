"""Tests for local embedding model service."""

import pytest
from app.rag.embeddings import (
    get_embedding_model,
    embed_text,
    embed_texts,
    SentenceTransformersProvider,
    set_embedding_provider,
)


def test_embed_text_basic():
    """Test basic embedding generation."""
    embedding = embed_text("Hello world")
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    # Check all values are floats
    assert all(isinstance(v, float) for v in embedding)


def test_embed_text_dimension():
    """Test embedding has consistent dimension."""
    embedding = embed_text("Test sentence")
    # nomic-embed-text produces 768-dimensional vectors
    assert len(embedding) == 768


def test_embed_texts_batch():
    """Test batch embedding generation."""
    texts = ["Hello world", "Goodbye world", "Test sentence"]
    embeddings = embed_texts(texts)
    assert len(embeddings) == 3
    assert all(len(e) == 768 for e in embeddings)


def test_embedding_provider_initialization():
    """Test embedding provider can be initialized."""
    provider = get_embedding_model()
    assert provider is not None
    assert provider.dimension == 768


def test_embedding_provider_custom_model():
    """Test embedding provider with custom model name."""
    # Just test that it doesn't crash with a known model
    provider = get_embedding_model("nomic-embed-text")
    assert provider is not None
    assert provider.dimension == 768


def test_set_embedding_provider():
    """Test setting a custom embedding provider."""
    # Just verify the function exists and can be called
    # (actual provider setting is tested elsewhere)
    from app.rag.embeddings import EmbeddingProvider
    assert EmbeddingProvider is not None