"""Document management endpoints."""
import os
import structlog
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from uuid import UUID
import uuid as uuid_module

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import (
    get_logger,
    log_audit_event,
    log_document_ingestion_event,
)
from app.core.logging import _sanitize_filename as sanitize_filename

logger = get_logger(__name__)
from app.models.document import Document, DocumentPermission, DocumentStatus
from app.models.user import User, DocumentClassification
from app.schemas.document import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentPermissionCreate,
    DocumentPermissionUpdate,
    DocumentPermissionResponse,
    DocumentReindexRequest,
)
from app.services.audit import audit_log
from app.api.v1.auth import get_current_user
from app.rag.service import RAGService
from app.rag.qdrant import QdrantManager

router = APIRouter(prefix="/documents", tags=["documents"])

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Check file size (read first chunk to estimate)
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size} exceeds maximum {settings.MAX_FILE_SIZE}"
        )


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    qdrant = QdrantManager(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.EMBEDDING_DIMENSION,
    )
    return RAGService(
        qdrant_manager=qdrant,
        embedding_model_name=settings.OLLAMA_EMBEDDING_MODEL,
        llm_model_name=settings.OLLAMA_MODEL,
        top_k=settings.TOP_K_RESULTS,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def process_document_background(
    document_id: str,
    file_path: str,
    original_filename: str,
    classification: str,
    department: str,
    user_role: str,
    uploaded_by_id: str,
):
    """Background task to process document through RAG pipeline.

    Runs in a separate thread/event loop to avoid blocking the API response.
    Pipeline: text extraction → chunking → embeddings → Qdrant vectors.
    """
    import asyncio
    import time as time_module
    from datetime import datetime
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.core.config import settings as config

    async def _process():
        engine = create_async_engine(config.DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        start_time = time_module.time()

        async with async_session() as session:
            # Get the document
            from app.models.document import Document, DocumentStatus, DocumentPermission

            result = await session.execute(
                select(Document).where(Document.id == uuid_module.UUID(document_id))
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.warning("Document not found for background processing", document_id=document_id)
                await engine.dispose()
                return

            # Mark as processing
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            await session.commit()

            try:
                # Get allowed_roles and allowed_users from document permissions
                perm_result = await session.execute(
                    select(DocumentPermission).where(DocumentPermission.document_id == document.id)
                )
                permissions = perm_result.scalars().all()

                # Build allowed_roles list from role-based permissions
                allowed_roles = []
                allowed_users = []
                for perm in permissions:
                    if perm.can_read:
                        if perm.role_id:
                            from app.models.user import Role
                            role_result = await session.execute(
                                select(Role).where(Role.id == perm.role_id)
                            )
                            role = role_result.scalar_one_or_none()
                            if role:
                                allowed_roles.append(role.name)
                        if perm.user_id:
                            allowed_users.append(str(perm.user_id))

                # Process through RAG pipeline
                rag_service = get_rag_service()
                ingestion_result = rag_service.process_document_upload(
                    file_path=file_path,
                    filename=original_filename,
                    user_role=user_role,
                    department=department,
                    classification=classification,
                    document_id=document_id,
                    uploaded_by_id=uploaded_by_id,
                    allowed_roles=list(set(allowed_roles)),  # deduplicate
                    allowed_users=list(set(allowed_users)),  # deduplicate
                )

                processing_duration_ms = int((time_module.time() - start_time) * 1000)

                # Update document with processing results
                if ingestion_result.success:
                    document.status = DocumentStatus.PROCESSED
                    document.page_count = ingestion_result.page_count
                    document.chunk_count = ingestion_result.chunk_count
                    document.processed_at = datetime.utcnow()
                    log_document_ingestion_event(
                        logger,
                        document_id=str(document.id),
                        filename=original_filename,
                        file_type=document.extension,
                        file_size=document.file_size,
                        page_count=ingestion_result.page_count,
                        chunk_count=ingestion_result.chunk_count,
                        classification=classification,
                        department=department,
                        status="processed",
                        processing_time_ms=processing_duration_ms,
                    )
                else:
                    document.status = DocumentStatus.FAILED
                    document.error_message = f"Processing failed: {ingestion_result.error}"
                    log_document_ingestion_event(
                        logger,
                        document_id=str(document.id),
                        filename=original_filename,
                        file_type=document.extension,
                        file_size=document.file_size,
                        page_count=ingestion_result.page_count,
                        chunk_count=ingestion_result.chunk_count,
                        classification=classification,
                        department=department,
                        status="failed",
                        processing_time_ms=processing_duration_ms,
                        error=ingestion_result.error,
                    )

                await session.commit()

            except Exception as e:
                document.status = DocumentStatus.FAILED
                error_category = type(e).__name__
                document.error_message = f"Processing failed: {error_category}"
                await session.commit()
                logger.error(
                    "document_ingestion_failed",
                    document_id=document_id,
                    error_category=error_category,
                )

            await engine.dispose()

    asyncio.run(_process())


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    classification: DocumentClassification = Form(DocumentClassification.PUBLIC_INTERNAL),
    department: str = Form(None),
    tags: str = Form(None),  # Comma-separated
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Upload a new document."""
    validate_file(file)

    # Generate unique filename
    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid_module.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get file size
    file_size = file_path.stat().st_size

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    # Create document record
    document = Document(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        mime_type=file.content_type or "application/octet-stream",
        extension=file_ext,
        title=title or file.filename,
        description=description,
        classification=classification,
        department=department,
        tags=tag_list,
        uploaded_by_id=current_user.id,
        department_id=current_user.department_id,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Add default permission for uploader
    permission = DocumentPermission(
        document_id=document.id,
        user_id=current_user.id,
        can_read=True,
        can_write=True,
        can_delete=True,
    )
    db.add(permission)

    # Add role-based permissions based on classification
    if classification == DocumentClassification.PUBLIC_INTERNAL:
        # All authenticated users can read
        from app.models.user import Role
        result = await db.execute(select(Role))
        roles = result.scalars().all()
        for role in roles:
            perm = DocumentPermission(
                document_id=document.id,
                role_id=role.id,
                can_read=True,
                can_write=False,
                can_delete=False,
            )
            db.add(perm)

    await db.commit()
    await db.refresh(document, ["permissions"])

    await audit_log(db, request, "document_uploaded", user_id=current_user.id,
                    resource_type="document", resource_id=document.id,
                    details={"filename": document.original_filename, "classification": classification.value})

    log_audit_event(
        logger,
        event_type="document_uploaded",
        user_id=str(current_user.id),
        resource_type="document",
        resource_id=str(document.id),
        details={
            "filename": document.original_filename,
            "classification": classification.value,
            "department": department,
            "file_size": file_size,
        },
    )

    # Trigger RAG processing in background
    background_tasks.add_task(
        process_document_background,
        document_id=str(document.id),
        file_path=str(file_path),
        original_filename=document.original_filename,
        classification=classification.value,
        department=department or "",
        user_role=current_user.role.name if current_user.role else "employee",
        uploaded_by_id=str(document.uploaded_by_id),
    )

    return document


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query("", description="Search by filename, title, or description"),
    classification: DocumentClassification = Query(None),
    status_filter: DocumentStatus = Query(None, alias="status"),
    department: str = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """List documents with pagination and filters."""
    # Base query with permission filter
    # Users can see documents they have read permission for
    subquery = select(DocumentPermission.document_id).where(
        (DocumentPermission.user_id == current_user.id) |
        (DocumentPermission.role_id == current_user.role_id)
    )

    query = select(Document).options(selectinload(Document.permissions)).where(
        Document.id.in_(subquery),
        Document.status != DocumentStatus.DELETED
    )

    if search:
        query = query.where(
            (Document.filename.ilike(f"%{search}%")) |
            (Document.title.ilike(f"%{search}%")) |
            (Document.description.ilike(f"%{search}%"))
        )
    if classification:
        query = query.where(Document.classification == classification)
    if status_filter:
        query = query.where(Document.status == status_filter)
    if department:
        query = query.where(Document.department == department)

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Pagination
    query = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(documents=documents, total=total, page=page, page_size=page_size)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    request: Request,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Get a specific document with permissions."""
    # Check permission
    perm_result = await db.execute(
        select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            ((DocumentPermission.user_id == current_user.id) |
             (DocumentPermission.role_id == current_user.role_id)),
            DocumentPermission.can_read == True
        )
    )
    if not perm_result.scalars().first():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this document")

    result = await db.execute(
        select(Document).options(selectinload(Document.permissions)).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await audit_log(db, request, "document_accessed", user_id=current_user.id,
                    resource_type="document", resource_id=document.id)

    return document


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    request: Request,
    document_id: UUID,
    document_data: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Update document metadata."""
    # Check write permission
    perm_result = await db.execute(
        select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            ((DocumentPermission.user_id == current_user.id) |
             (DocumentPermission.role_id == current_user.role_id)),
            DocumentPermission.can_write == True
        )
    )
    if not perm_result.scalar_one_or_none() and current_user.role.name != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to modify this document")

    result = await db.execute(
        select(Document).options(selectinload(Document.permissions)).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    update_data = document_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(document, field, value)

    await db.commit()
    await db.refresh(document, ["permissions"])

    await audit_log(db, request, "document_updated", user_id=current_user.id,
                    resource_type="document", resource_id=document.id)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    request: Request,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document (soft delete)."""
    # Check delete permission
    perm_result = await db.execute(
        select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            ((DocumentPermission.user_id == current_user.id) |
             (DocumentPermission.role_id == current_user.role_id)),
            DocumentPermission.can_delete == True
        )
    )
    if not perm_result.scalar_one_or_none() and current_user.role.name != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this document")

    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Soft-delete the database record and remove its vectors together. Without
    # this, an old Qdrant point could remain retrievable after authorization
    # to the document has been revoked.
    qdrant = QdrantManager(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.EMBEDDING_DIMENSION,
    )
    if not qdrant.delete_document_vectors(document.id):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document vectors could not be removed; document was not deleted",
        )

    document.status = DocumentStatus.DELETED
    await db.commit()

    await audit_log(db, request, "document_deleted", user_id=current_user.id,
                    resource_type="document", resource_id=document.id)


@router.post("/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(
    request: Request,
    background_tasks: BackgroundTasks,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger document re-indexing through the RAG pipeline.

    Re-processes the document: text extraction → chunking → embeddings → Qdrant vectors.
    Existing vectors for the document are deleted before re-indexing.
    """
    if current_user.role.name not in ["admin", "manager"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin or manager access required")

    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Update status to processing
    document.status = DocumentStatus.PROCESSING
    document.error_message = None
    await db.commit()

    # Trigger re-indexing via background task
    background_tasks.add_task(
        process_document_background,
        document_id=str(document.id),
        file_path=document.file_path,
        original_filename=document.original_filename,
        classification=document.classification.value,
        department=document.department or "",
        user_role=current_user.role.name if current_user.role else "employee",
        uploaded_by_id=str(document.uploaded_by_id),
    )

    await audit_log(db, request, "document_reindex_requested", user_id=current_user.id,
                    resource_type="document", resource_id=document.id,
                    details={"document_id": str(document_id), "filename": document.original_filename})

    return {"message": "Document re-indexing initiated", "document_id": str(document_id)}


# Document Permission endpoints
@router.post("/{document_id}/permissions", response_model=DocumentPermissionResponse, status_code=status.HTTP_201_CREATED)
async def add_document_permission(
    request: Request,
    document_id: UUID,
    perm_data: DocumentPermissionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentPermissionResponse:
    """Add permission to a document (admin or document owner)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role.name != "admin" and document.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    # Validate either role_id or user_id is provided
    if not perm_data.role_id and not perm_data.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either role_id or user_id must be provided")

    permission = DocumentPermission(
        document_id=document_id,
        role_id=perm_data.role_id,
        user_id=perm_data.user_id,
        can_read=perm_data.can_read,
        can_write=perm_data.can_write,
        can_delete=perm_data.can_delete,
    )
    db.add(permission)
    await db.commit()
    await db.refresh(permission)

    await audit_log(db, request, "document_permission_changed", user_id=current_user.id,
                    resource_type="document", resource_id=document_id,
                    details={"permission": perm_data.model_dump()})

    return permission


@router.get("/{document_id}/permissions", response_model=List[DocumentPermissionResponse])
async def list_document_permissions(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[DocumentPermissionResponse]:
    """List permissions for a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role.name != "admin" and document.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    result = await db.execute(
        select(DocumentPermission).where(DocumentPermission.document_id == document_id)
    )
    return result.scalars().all()


@router.patch("/{document_id}/permissions/{permission_id}", response_model=DocumentPermissionResponse)
async def update_document_permission(
    request: Request,
    document_id: UUID,
    permission_id: UUID,
    perm_data: DocumentPermissionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentPermissionResponse:
    """Update a document permission."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role.name != "admin" and document.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    result = await db.execute(
        select(DocumentPermission).where(
            DocumentPermission.id == permission_id,
            DocumentPermission.document_id == document_id
        )
    )
    permission = result.scalar_one_or_none()
    if not permission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

    update_data = perm_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(permission, field, value)

    await db.commit()
    await db.refresh(permission)

    await audit_log(db, request, "document_permission_changed", user_id=current_user.id,
                    resource_type="document", resource_id=document_id,
                    details={"permission_id": str(permission_id), "changes": update_data})

    return permission


@router.delete("/{document_id}/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_permission(
    request: Request,
    document_id: UUID,
    permission_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document permission."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.role.name != "admin" and document.uploaded_by_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    result = await db.execute(
        select(DocumentPermission).where(
            DocumentPermission.id == permission_id,
            DocumentPermission.document_id == document_id
        )
    )
    permission = result.scalar_one_or_none()
    if not permission:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

    await db.delete(permission)
    await db.commit()

    await audit_log(db, request, "document_permission_changed", user_id=current_user.id,
                    resource_type="document", resource_id=document_id,
                    details={"permission_id": str(permission_id), "action": "deleted"})
