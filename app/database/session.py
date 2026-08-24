from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from app.database.engine import engine
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

