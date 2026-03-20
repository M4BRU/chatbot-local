"""SQLAlchemy async engine, session factory et base déclarative."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config.settings import Settings

_settings = Settings()

engine = create_async_engine(
    _settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency FastAPI — fournit une session async par requête."""
    async with AsyncSessionLocal() as session:
        yield session
