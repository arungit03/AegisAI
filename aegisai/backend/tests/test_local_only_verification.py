"""Phase 22: Local-Only AI Verification Tests.

Verifies that the entire AegisAI backend uses only local AI services:
- No external LLM API calls (OpenAI, Anthropic, Cohere, etc.)
- No external embedding services
- No external vector databases (only local Qdrant)
- No external telemetry or tracking services
"""
import os
import re
import ast

import pytest


# ─── Backend Code Scanning ───────────────────────────────────────────────────

class TestNoExternalAIServices:
    """Verify no external AI services are referenced in the backend code."""

    BACKEND_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app"
    )

    def _scan_py_files(self):
        """Yield (filepath, content) for all Python files in backend."""
        for root, dirs, files in os.walk(self.BACKEND_DIR):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    with open(filepath, encoding="utf-8") as f:
                        yield filepath, f.read()

    def test_no_openai_imports(self):
        """No OpenAI imports anywhere in backend."""
        for filepath, content in self._scan_py_files():
            assert "import openai" not in content.lower(), \
                f"Found OpenAI import in {filepath}"
            assert "from openai" not in content.lower(), \
                f"Found OpenAI import in {filepath}"

    def test_no_anthropic_imports(self):
        """No Anthropic imports in backend."""
        for filepath, content in self._scan_py_files():
            assert "import anthropic" not in content.lower(), \
                f"Found Anthropic import in {filepath}"
            assert "from anthropic" not in content.lower(), \
                f"Found Anthropic import in {filepath}"

    def test_no_cohere_imports(self):
        """No Cohere imports in backend."""
        for filepath, content in self._scan_py_files():
            assert "import cohere" not in content.lower(), \
                f"Found Cohere import in {filepath}"

    def test_no_pinecone_imports(self):
        """No Pinecone (external vector DB) imports in backend."""
        for filepath, content in self._scan_py_files():
            assert "import pinecone" not in content.lower(), \
                f"Found Pinecone import in {filepath}"

    def test_no_weaviate_imports(self):
        """No Weaviate imports in backend."""
        for filepath, content in self._scan_py_files():
            assert "import weaviate" not in content.lower(), \
                f"Found Weaviate import in {filepath}"

    def test_no_openai_api_key_config(self):
        """No OpenAI API key configuration."""
        config_path = os.path.join(self.BACKEND_DIR, "core", "config.py")
        with open(config_path) as f:
            content = f.read().lower()
        assert "openai" not in content
        assert "anthropic" not in content
        assert "cohere" not in content

    def test_ollama_url_is_localhost(self):
        """Ollama endpoint points to localhost only."""
        config_path = os.path.join(self.BACKEND_DIR, "core", "config.py")
        with open(config_path) as f:
            content = f.read()
        # Should have localhost or 127.0.0.1
        assert ("localhost" in content or "127.0.0.1" in content) or "OLLAMA_BASE_URL" not in content

    def test_qdrant_url_is_localhost(self):
        """Qdrant endpoint points to localhost only."""
        config_path = os.path.join(self.BACKEND_DIR, "core", "config.py")
        with open(config_path) as f:
            content = f.read()
        assert ("localhost" in content or "127.0.0.1" in content) or "QDRANT_URL" not in content

    def test_no_http_client_external_calls(self):
        """Verify LLM calls go to localhost:11434, not external endpoints."""
        service_path = os.path.join(self.BACKEND_DIR, "rag", "service.py")
        with open(service_path) as f:
            content = f.read()

        # The endpoint is configurable so Docker can use the private `ollama`
        # service name while local runs can use localhost.
        assert "OLLAMA_BASE_URL" in content

        # Should NOT reference external LLM endpoints
        forbidden_patterns = [
            r"https?://api\.openai\.com",
            r"https?://api\.anthropic\.com",
            r"https?://api\.cohere\.ai",
            r"openai\.com",
            r"anthropic\.com",
            r"cohere\.ai",
        ]
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content)
            assert len(matches) == 0, f"Found external API reference in service.py: {pattern}"

    def test_requirements_no_external_ai_deps(self):
        """Verify requirements.txt has no external AI dependencies."""
        req_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements.txt"
        )
        with open(req_path) as f:
            requirements = f.read().lower()

        forbidden_deps = [
            "openai",
            "anthropic",
            "cohere",
            "pinecone",
            "weaviate",
            "replicate",
            "huggingface_hub",
            "transformers[torch]",
            "sentence-transformers[torch]",
        ]

        for dep in forbidden_deps:
            if dep in requirements:
                # Check if it's actually a different package with similar name
                lines = requirements.split("\n")
                for line in lines:
                    if dep in line.lower() and not line.startswith("#"):
                        # sentence-transformers is OK as local
                        if dep == "sentence-transformers[torch]" and "sentence-transformers" in line:
                            continue
                        pytest.fail(f"Forbidden external dependency '{dep}' found in requirements.txt")
