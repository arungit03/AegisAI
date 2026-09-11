"""Health check endpoints."""
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.config import settings
from app.core.database import get_db, AsyncSessionLocal
from app.schemas.health import HealthResponse, SystemStatusResponse, ServiceStatus
from app.rag.qdrant import QdrantManager

router = APIRouter(prefix="/health", tags=["health"])

# Track startup time for uptime calculation
startup_time = time.time()


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Readiness check - verifies database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return HealthResponse(
            status="ready",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
        )
    except Exception:
        return HealthResponse(
            status="not_ready",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
        )


@router.get("/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    """Liveness check - verifies service is running."""
    return HealthResponse(
        status="alive",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/qdrant")
async def qdrant_health():
    """Qdrant-specific health check with collection info.

    Returns:
        Dict with Qdrant health status and collection statistics.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.QDRANT_URL}/health")
            if response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "error": f"Qdrant returned status {response.status_code}",
                }

            # Get collection info
            try:
                qdrant = QdrantManager(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vector_size=settings.EMBEDDING_DIMENSION,
                )
                info = qdrant.get_collection_info()
                response_data = {
                    "status": "healthy",
                    "url": settings.QDRANT_URL,
                    "collection": settings.QDRANT_COLLECTION_NAME,
                }
                if info:
                    response_data["vectors_count"] = info.get("vectors_count", 0)
                    response_data["config"] = info.get("config", {})
                return response_data
            except Exception as collection_err:
                return {
                    "status": "healthy",
                    "url": settings.QDRANT_URL,
                    "collection": settings.QDRANT_COLLECTION_NAME,
                    "collection_error": str(collection_err),
                }

    except httpx.TimeoutException:
        return {"status": "unhealthy", "error": "Qdrant request timed out"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(db: AsyncSession = Depends(get_db)) -> SystemStatusResponse:
    """Detailed system status with service checks."""
    services: Dict[str, ServiceStatus] = {}
    overall_status = "healthy"

    # Check Database
    db_start = time.time()
    try:
        await db.execute(text("SELECT 1"))
        db_latency = (time.time() - db_start) * 1000
        services["database"] = ServiceStatus(
            name="PostgreSQL",
            status="healthy",
            latency_ms=db_latency,
            details={"url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "configured"}
        )
    except Exception as e:
        services["database"] = ServiceStatus(
            name="PostgreSQL",
            status="unhealthy",
            details={"error": str(e)}
        )
        overall_status = "unhealthy"

    # Check Qdrant
    qdrant_start = time.time()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.QDRANT_URL}/health")
            qdrant_latency = (time.time() - qdrant_start) * 1000
            if response.status_code == 200:
                services["qdrant"] = ServiceStatus(
                    name="Qdrant Vector DB",
                    status="healthy",
                    latency_ms=qdrant_latency,
                    details={"url": settings.QDRANT_URL}
                )
            else:
                services["qdrant"] = ServiceStatus(
                    name="Qdrant Vector DB",
                    status="degraded",
                    latency_ms=qdrant_latency,
                    details={"status_code": response.status_code}
                )
                overall_status = "degraded"
    except Exception as e:
        services["qdrant"] = ServiceStatus(
            name="Qdrant Vector DB",
            status="unhealthy",
            details={"error": str(e)}
        )
        overall_status = "unhealthy"

    # Check Ollama
    ollama_start = time.time()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_latency = (time.time() - ollama_start) * 1000
            if response.status_code == 200:
                services["ollama"] = ServiceStatus(
                    name="Ollama LLM",
                    status="healthy",
                    latency_ms=ollama_latency,
                    details={"url": settings.OLLAMA_BASE_URL, "model": settings.OLLAMA_MODEL}
                )
            else:
                services["ollama"] = ServiceStatus(
                    name="Ollama LLM",
                    status="degraded",
                    latency_ms=ollama_latency,
                    details={"status_code": response.status_code}
                )
                overall_status = "degraded"
    except Exception as e:
        services["ollama"] = ServiceStatus(
            name="Ollama LLM",
            status="unhealthy",
            details={"error": str(e)}
        )
        overall_status = "unhealthy"

    uptime = time.time() - startup_time

    return SystemStatusResponse(
        status=overall_status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        services=services,
        uptime_seconds=uptime,
    )