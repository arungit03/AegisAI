"""Health check and system status schemas."""
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Basic health check response."""
    status: str
    service: str
    version: str


class ServiceStatus(BaseModel):
    """Individual service status."""
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SystemStatusResponse(BaseModel):
    """Detailed system status response."""
    status: str
    service: str
    version: str
    environment: str
    services: Dict[str, ServiceStatus]
    uptime_seconds: float