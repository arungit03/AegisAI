"""Database configuration and session management."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.core.config import settings
from app.core.security import get_password_hash


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# SQLite doesn't support pool_size/max_overflow; only apply for PostgreSQL
_engine_kwargs = {
    "echo": False,  # Security: Never log SQL statements (may contain user data)
    "future": True,
}
if "sqlite" not in settings.DATABASE_URL:
    _engine_kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
    _engine_kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW

engine = create_async_engine(
    settings.DATABASE_URL,
    **_engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables and seed default data."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_db()


async def seed_db() -> None:
    """Seed default roles, departments, and admin user."""
    async with AsyncSessionLocal() as session:
        from app.models.user import User, Role, Department

        # Seed roles
        for role_name in ["admin", "manager", "engineer", "employee"]:
            result = await session.execute(
                select(Role).where(Role.name == role_name)
            )
            if not result.scalar_one_or_none():
                session.add(Role(name=role_name))

        # Seed default department
        result = await session.execute(
            select(Department).where(Department.name == "Engineering")
        )
        if not result.scalar_one_or_none():
            session.add(Department(name="Engineering", description="Default department"))

        await session.commit()

        # Seed admin user
        result = await session.execute(
            select(User).where(User.username == "admin")
        )
        if not result.scalar_one_or_none():
            result = await session.execute(
                select(Role).where(Role.name == "admin")
            )
            admin_role = result.scalar_one()

            result = await session.execute(
                select(Department).where(Department.name == "Engineering")
            )
            eng_dept = result.scalar_one()

            admin_user = User(
                email="admin@aegisai.com",
                username="admin",
                hashed_password=get_password_hash("admin123"),
                full_name="Admin User",
                is_active=True,
                is_superuser=True,
                role_id=admin_role.id,
                department_id=eng_dept.id,
            )
            session.add(admin_user)

        await session.commit()


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()