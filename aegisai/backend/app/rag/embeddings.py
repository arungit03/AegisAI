"""Local embedding model service for RAG.

Abstracts embedding model provider so it can be changed later.
Currently supports sentence-transformers models running locally.

All embedding operations stay local - no external APIs.
"""

import structlog
from typing import List, Optional, Tuple
import numpy as np
import httpx

from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingProvider:
    """Abstract base class for embedding providers."""

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings through the configured Ollama service."""

    def __init__(self, model_name: str, base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._dimension = settings.EMBEDDING_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                raise RuntimeError("Ollama returned an empty embedding")
            self._dimension = len(embedding)
            return embedding
        except Exception as e:
            logger.error("Ollama embedding generation failed", error_type=type(e).__name__)
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence-Transformers model running locally.

    Uses sentence-transformers library with models that run offline.
    No external API calls after model download.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model.
                "nomic-embed-text" is a good default for general use.
            device: Computing device ("cpu" or "cuda").
            cache_dir: Directory to cache the model.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None

        logger.info("Loading embedding model", device=device)

        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded", dimension=self._dimension)
        except Exception as e:
            logger.error("Failed to load embedding model", error_type=type(e).__name__)
            raise

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        if not self._model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error("Embedding generation failed", error_type=type(e).__name__)
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self._model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Batch embedding generation failed", error_type=type(e).__name__)
            raise


# Global provider instance
_embedding_provider: Optional[EmbeddingProvider] = None


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingProvider:
    """Get or create the global embedding model instance.

    Args:
        model_name: Name of the model to use.
            If None, uses the default "nomic-embed-text".

    Returns:
        EmbeddingProvider instance.
    """
    global _embedding_provider

    embedding_model_name = model_name or settings.OLLAMA_EMBEDDING_MODEL

    if _embedding_provider is None or _embedding_provider.model_name != embedding_model_name:
        if _embedding_provider and _embedding_provider.model_name == embedding_model_name:
            # Already have the right model
            pass
        else:
            if embedding_model_name == settings.OLLAMA_EMBEDDING_MODEL:
                _embedding_provider = OllamaEmbeddingProvider(embedding_model_name)
            else:
                _embedding_provider = SentenceTransformersProvider(embedding_model_name)

    return _embedding_provider


def set_embedding_provider(provider: EmbeddingProvider) -> None:
    """Set a custom embedding provider.

    Args:
        provider: An EmbeddingProvider instance.
    """
    global _embedding_provider
    _embedding_provider = provider


def embed_text(text: str, model_name: Optional[str] = None) -> List[float]:
    """Convenience function to embed a single text.

    Args:
        text: The text to embed.
        model_name: Optional model name override.

    Returns:
        Embedding vector as list of floats.
    """
    provider = get_embedding_model(model_name)
    return provider.embed(text)


def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """Convenience function to embed multiple texts.

    Args:
        texts: List of texts to embed.
        model_name: Optional model name override.

    Returns:
        List of embedding vectors.
    """
    provider = get_embedding_model(model_name)
    return provider.embed_batch(texts)
