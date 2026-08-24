from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

