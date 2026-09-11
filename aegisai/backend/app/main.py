"""Main FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
import uuid as uuid_module

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.logging import configure_logging, get_logger
from app.api.v1 import health, auth, users, documents, chat, audit


# Configure structured logging
logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting AegisAI", version=settings.APP_VERSION, environment=settings.ENVIRONMENT)
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down AegisAI")
    await close_db()
    logger.info("Database connections closed")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Private, secure, enterprise-grade AI assistant for Smart India Hackathon",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with security-conscious filtering.

    Does NOT log:
    - Query parameters (may contain tokens, PII, or sensitive search terms)
    - Request body
    - Authorization headers
    - Full URLs with query strings

    Logs only: method, path (without query string), client IP, request_id
    """
    start_time = time.time()

    # Generate request ID for correlation
    request_id = str(uuid_module.uuid4())
    request.state.request_id = request_id

    # Log only the path without query parameters to avoid leaking sensitive data
    path = request.url.path
    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=path,
        client_ip=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=path,
            status_code=response.status_code,
            process_time_ms=round(process_time * 1000, 2),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        # Log error category only, not the full exception message
        logger.error(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=path,
            error_category=type(e).__name__,
            process_time_ms=round(process_time * 1000, 2),
        )
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler.

    Logs error category only (not full message) to prevent leaking internals.
    Returns a generic error to the client.
    """
    request_id = getattr(request.state, "request_id", None)
    path = request.url.path
    error_category = type(exc).__name__
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        path=path,
        error_category=error_category,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(audit.router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs" if settings.DEBUG else "disabled",
    }


