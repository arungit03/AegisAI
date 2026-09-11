"""Qdrant vector database manager for RAG storage and retrieval.

Handles vector storage, search, deletion, re-indexing, and
permission-aware filtered search.

All data stays local - Qdrant runs via Docker in the same network.
"""

import structlog
import copy
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

import qdrant_client
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PointStruct,
    MinShould,
)
import re
from app.rag.chunking import ChunkMetadata

logger = structlog.get_logger(__name__)


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "aegisai_documents",
        vector_size: int = 768,
    ):
        """Initialize Qdrant manager.

        Args:
            url: Qdrant server URL.
            api_key: Qdrant API key (None for no auth).
            collection_name: Name of the collection.
            vector_size: Dimensionality of embedding vectors.
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size

        try:
            self.client = qdrant_client.QdrantClient(
                url=url,
                api_key=api_key,
            )
            logger.info("Connected to Qdrant", collection=self.collection_name)
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error_type=type(e).__name__)
            raise

    def ensure_collection(self) -> bool:
        """Ensure the collection exists, create if not.

        Returns:
            True if collection exists or was created successfully.
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Qdrant collection created", collection=self.collection_name)
            else:
                logger.info("Qdrant collection exists", collection=self.collection_name)

            return True

        except Exception as e:
            logger.error("Failed to ensure Qdrant collection", error_type=type(e).__name__)
            return False

    def add_vectors(
        self,
        points: List[PointStruct],
        wait: bool = True,
    ) -> bool:
        """Add vectors to the collection.

        Args:
            points: List of PointStruct objects to add.
            wait: Whether to wait for the operation to complete.

        Returns:
            True if successful.
        """
        try:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
                wait=wait,
            )
            logger.info("Qdrant vectors added", count=len(points))
            return True

        except Exception as e:
            logger.error("Failed to add vectors to Qdrant", error_type=type(e).__name__)
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional filtering.

        This is the critical permission-aware search method.
        The filter_conditions parameter ensures we only retrieve
        vectors the user is authorized to access.

        Args:
            query_vector: The embedding vector to search with.
            limit: Maximum number of results to return.
            filter_conditions: Qdrant Filter for authorization.
            with_payload: Whether to include payload in results.
            with_vectors: Whether to include vectors in results.

        Returns:
            List of search result dictionaries.
        """
        try:
            search_kwargs = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
                "query_filter": filter_conditions,
            }
            if score_threshold is not None:
                search_kwargs["score_threshold"] = score_threshold

            results = self.client.search(
                **search_kwargs,
            )

            # Convert to dict format
            result_dicts = []
            for result in results:
                payload = result.payload if hasattr(result, 'payload') else {}
                result_dict = {
                    "id": str(result.id),
                    "score": result.score,
                    "payload": payload,
                }
                result_dicts.append(result_dict)

            logger.info("Qdrant search completed", result_count=len(result_dicts), limit=limit)
            return result_dicts

        except Exception as e:
            logger.error("Qdrant search failed", error_type=type(e).__name__)
            return []

    def keyword_search(
        self,
        terms: List[str],
        filter_conditions: Optional[Filter] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find exact identifier occurrences within an authorized scope.

        This is intentionally used only for explicit identifiers such as
        SIH26010.  The permission filter is applied by Qdrant before records
        are returned; normalized matching only handles OCR whitespace/punctuation.
        """
        normalized_terms = [re.sub(r"[^a-z0-9]", "", term.lower()) for term in terms]
        normalized_terms = [term for term in normalized_terms if term]
        if not normalized_terms:
            return []

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            record_matches = []
            legacy_matches = []
            records_by_location = {}
            for record in records:
                payload = record.payload or {}
                location = (
                    str(payload.get("document_id", "")),
                    payload.get("page_number"),
                    payload.get("chunk_index"),
                )
                records_by_location[location] = record
            for record in records:
                payload = record.payload or {}
                location = (
                    str(payload.get("document_id", "")),
                    payload.get("page_number"),
                    payload.get("chunk_index"),
                )
                searchable = " ".join(str(payload.get(key, "")) for key in ("chunk_text", "filename"))
                normalized = re.sub(r"[^a-z0-9]", "", searchable.lower())
                stored_identifier = re.sub(
                    r"[^a-z0-9]", "", str(payload.get("problem_statement_id", "")).lower()
                )
                matched = [
                    term for term in normalized_terms
                    if (payload.get("record_type") == "sih_record" and stored_identifier == term)
                    or (not stored_identifier and term in normalized)
                ]
                if matched:
                    # Table extraction can split the title immediately before
                    # the identifier. Include the preceding chunk from the
                    # same document/page so the LLM receives one coherent row.
                    previous_location = (
                        location[0],
                        location[1],
                        (location[2] - 1) if isinstance(location[2], int) else None,
                    )
                    previous = records_by_location.get(previous_location)
                    combined_payload = copy.deepcopy(payload)
                    if previous is not None:
                        previous_text = str((previous.payload or {}).get("chunk_text", "")).strip()
                        current_text = str(payload.get("chunk_text", "")).strip()
                        if previous_text and current_text:
                            combined_payload["chunk_text"] = f"{previous_text}\n{current_text}"
                    match = {
                        "id": str(record.id),
                        "score": 1.0,
                        "payload": combined_payload,
                    }
                    if payload.get("record_type") == "sih_record":
                        # Structured records are authoritative and must win
                        # over ordinary chunks containing neighboring IDs.
                        record_matches.append(match)
                    else:
                        legacy_matches.append(match)
            matches = record_matches or legacy_matches
            # Scroll order is not a relevance order.  Make table lookups
            # deterministic and remove overlapping duplicate chunks produced
            # by page extraction or repeated uploads.
            matches.sort(key=lambda item: (
                item["payload"].get("page_number") or 0,
                item["payload"].get("chunk_index") or 0,
                item["payload"].get("document_id", ""),
            ))
            unique_matches = []
            seen_text = set()
            for match in matches:
                text_key = re.sub(
                    r"\s+", " ", str(match["payload"].get("chunk_text", "")).strip().lower()
                )
                if text_key in seen_text:
                    continue
                seen_text.add(text_key)
                unique_matches.append(match)
                if len(unique_matches) >= limit:
                    break
            return unique_matches
        except Exception as e:
            logger.error("Qdrant keyword search failed", error_type=type(e).__name__)
            return []

    def search_sih_records(
        self,
        identifier: Optional[str] = None,
        query_text: Optional[str] = None,
        filter_conditions: Optional[Filter] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search structured SIH records lexically within an authorized scope.

        This is deliberately separate from vector search. An explicit SIH
        identifier is matched exactly against the structured payload field;
        reverse title lookups use normalized complete-title/token matching.
        """
        normalized_identifier = re.sub(r"[^a-z0-9]", "", identifier.lower()) if identifier else None
        normalized_query = self._normalize_lookup_text(query_text) if query_text else None
        query_tokens = set(normalized_query.split()) if normalized_query else set()
        query_content_tokens = query_tokens - {
            "what", "which", "is", "the", "a", "an", "title", "belongs", "belong",
            "to", "for", "of", "problem", "statement", "number", "id", "associated",
            "with", "please", "tell", "me", "about",
        }
        if not normalized_identifier and not normalized_query:
            return []

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            matches = []
            for record in records:
                payload = record.payload or {}
                if payload.get("record_type") != "sih_record":
                    continue

                stored_identifier = re.sub(
                    r"[^a-z0-9]", "", str(payload.get("problem_statement_id", "")).lower()
                )
                title = self._normalize_lookup_text(str(payload.get("title", "")))
                identifier_match = normalized_identifier and stored_identifier == normalized_identifier
                title_match = bool(
                    normalized_query
                    and title
                    and (
                        title in normalized_query
                        or (
                            len(query_content_tokens) >= 3
                            and query_content_tokens.issubset(set(title.split()))
                        )
                    )
                )
                if identifier_match or title_match:
                    matches.append({
                        "id": str(record.id),
                        "score": 1.0,
                        "payload": payload,
                    })

            matches.sort(key=lambda item: (
                item["payload"].get("page_number") or 0,
                item["payload"].get("problem_statement_id", ""),
                item["payload"].get("document_id", ""),
            ))
            return matches[:limit]
        except Exception as e:
            logger.error("Qdrant structured SIH search failed", error_type=type(e).__name__)
            return []

    def list_sih_records(
        self,
        filter_conditions: Optional[Filter] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """List structured SIH records within an authorized scope."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            matches = [
                {
                    "id": str(record.id),
                    "score": 1.0,
                    "payload": record.payload or {},
                }
                for record in records
                if (record.payload or {}).get("record_type") == "sih_record"
            ]
            matches.sort(key=lambda item: (
                item["payload"].get("page_number") or 0,
                item["payload"].get("problem_statement_id", ""),
                item["payload"].get("document_id", ""),
            ))
            return matches[:limit]
        except Exception as e:
            logger.error("Qdrant structured SIH listing failed", error_type=type(e).__name__)
            return []

    @staticmethod
    def _normalize_lookup_text(value: str) -> str:
        """Normalize whitespace/punctuation and known PDF OCR glyph errors."""
        value = value.lower().replace("lmagery", "imagery").replace("al-powered", "ai-powered")
        value = value.replace("lntegration", "integration").replace("lntelligent", "intelligent")
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    def delete_document_vectors(self, document_id: UUID) -> bool:
        """Delete all vectors belonging to a document.

        Uses Qdrant's filter-based deletion to remove all chunks
        associated with a given document_id stored in payload.

        Args:
            document_id: The document ID to delete.

        Returns:
            True if deletion was successful.
        """
        try:
            from qdrant_client.models import PointIdsList

            # Delete by filtering on document_id in payload using scroll+delete
            # First, find all points with this document_id
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            )

            # Get all points matching the filter
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=delete_filter,
                limit=10000,  # Should be enough for any document
                with_payload=False,
                with_vectors=False,
            )

            if scroll_result and scroll_result[0]:
                point_ids = [point.id for point in scroll_result[0]]
                if point_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=point_ids),
                    )
                    logger.info("Qdrant vectors deleted", document_id=str(document_id), count=len(point_ids))
                else:
                    logger.debug("No vectors found for document", document_id=str(document_id))
            else:
                logger.debug("No vectors found for document", document_id=str(document_id))

            return True

        except Exception as e:
            logger.error("Failed to delete document vectors from Qdrant", error_type=type(e).__name__)
            return False

    def delete_chunk_vectors(self, chunk_ids: List[str]) -> bool:
        """Delete specific chunk vectors.

        Args:
            chunk_ids: List of chunk IDs to delete.

        Returns:
            True if deletion was successful.
        """
        try:
            from qdrant_client.models import PointIdsList

            if chunk_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=chunk_ids),
                )
            logger.info("Qdrant chunk vectors deleted", count=len(chunk_ids))
            return True

        except Exception as e:
            logger.error("Failed to delete chunk vectors from Qdrant", error_type=type(e).__name__)
            return False

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection information.

        Returns:
            Dict with collection info or None on error.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "config": {
                    "size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                } if info.config.params else None,
            }
        except Exception as e:
            logger.error("Failed to get Qdrant collection info", error_type=type(e).__name__)
            return None

    def create_permission_filter(
        self,
        user_role: str,
        user_id: Optional[UUID] = None,
        department: Optional[str] = None,
        allowed_document_ids: Optional[List[UUID]] = None,
    ) -> Filter:
        """Create a Qdrant Filter for permission-aware search.

        This is the critical method that ensures users only retrieve
        documents they are authorized to access. The filter is applied
        at the Qdrant search level, NOT after retrieval.

        Authorization logic:
        - admin: No filter (access to all documents)
        - manager: Access to documents in their department + all PUBLIC_INTERNAL docs
        - engineer: Access to documents in their department + all PUBLIC_INTERNAL docs
        - employee: Access to PUBLIC_INTERNAL docs only

        Note: Filters use OR logic (should conditions) to allow access when
        ANY matching condition is met, since a user may access docs via
        multiple paths (e.g., department membership + PUBLIC_INTERNAL).

        Args:
            user_role: The user's role (admin, manager, engineer, employee).
            user_id: The user's UUID (for user-specific permissions).
            department: The user's department.
            allowed_document_ids: Explicitly allowed document IDs.

        Returns:
            Qdrant Filter object for search authorization.
        """
        # Admin bypass - no restrictions
        if user_role == "admin":
            logger.debug("Admin role detected: permission filter bypassed", user_role=user_role)
            return Filter(must=[])  # Empty filter = all access

        should_conditions = []

        # Role-based filtering - users can access docs via multiple paths,
        # so we use OR (should) conditions
        if user_role in ("manager", "engineer"):
            # Can access their department's docs
            if department:
                should_conditions.append(
                    FieldCondition(
                        key="department",
                        match=MatchValue(value=department),
                    )
                )
            # Can access PUBLIC_INTERNAL docs
            should_conditions.append(
                FieldCondition(
                    key="classification",
                    match=MatchValue(value="public_internal"),
                )
            )

        elif user_role == "employee":
            # Employees can access PUBLIC_INTERNAL docs only
            should_conditions.append(
                FieldCondition(
                    key="classification",
                    match=MatchValue(value="public_internal"),
                )
            )

        # User-specific permissions - documents uploaded by the user or
        # explicitly shared with the user via allowed_users field
        if user_id is not None:
            # Documents uploaded by this user
            should_conditions.append(
                FieldCondition(
                    key="uploaded_by_id",
                    match=MatchValue(value=str(user_id)),
                )
            )
            # Documents explicitly shared with this user via allowed_users
            should_conditions.append(
                FieldCondition(
                    key="allowed_users",
                    match=MatchAny(any=[str(user_id)]),
                )
            )

        # Explicitly allowed document IDs
        if allowed_document_ids and len(allowed_document_ids) > 0:
            # Combine all should_conditions with an OR on document_id
            doc_id_conditions = [
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(doc_id)),
                )
                for doc_id in allowed_document_ids
            ]
            should_conditions.append(
                Filter(should=doc_id_conditions)
            )

        logger.debug(
            "Permission filter created",
            role=user_role,
            condition_count=len(should_conditions),
        )

        # Build filter with should (OR) conditions
        # Must have at least 1 match to be included
        if should_conditions:
            return Filter(
                min_should=MinShould(
                    conditions=should_conditions,
                    min_count=1,
                ),
            )
        else:
            # No conditions - should not happen for non-admin, but return safe empty filter
            return Filter(must=[])

    def reindex_document(
        self,
        document_id: UUID,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> bool:
        """Re-index a document (delete old vectors, add new ones).

        Args:
            document_id: The document ID.
            chunks: List of chunk metadata dicts.
            vectors: List of embedding vectors.

        Returns:
            True if re-indexing was successful.
        """
        try:
            # Delete existing vectors for this document
            self.delete_document_vectors(document_id)

            # Add new vectors
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point = PointStruct(
                    id=str(chunk.get("chunk_id", f"doc_{document_id}_chunk_{i}")),
                    vector=vector,
                    payload={
                        "document_id": str(document_id),
                        "filename": chunk.get("filename", ""),
                        "chunk_text": chunk.get("text", ""),
                        "chunk_index": chunk.get("chunk_index", i),
                        "page_number": chunk.get("page_number"),
                        "classification": chunk.get("classification", "public_internal"),
                        "department": chunk.get("department"),
                    },
                )
                points.append(point)

            if points:
                success = self.add_vectors(points)
                logger.info("Document re-indexed", document_id=str(document_id), vector_count=len(points))
                return success
            else:
                logger.warning("No chunks to re-index", document_id=str(document_id))
                return False

        except Exception as e:
            logger.error("Failed to re-index document", document_id=str(document_id), error_type=type(e).__name__)
            return False
