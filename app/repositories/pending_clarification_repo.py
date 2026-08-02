
from datetime import datetime, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.pending_clarification import PendingClarification

TTL_HOURS = 24


async def save_pending_clarification(
    session: AsyncSession, conversation_id: str, category_id: str
) -> None:
    existing = await session.get(PendingClarification, conversation_id)
    if existing:
        existing.category_id = category_id
        existing.created_at = datetime.utcnow()
    else:
        session.add(PendingClarification(
            conversation_id=conversation_id,
            category_id=category_id,
            created_at=datetime.utcnow(),
        ))
    await session.commit()


async def get_pending_clarification(
    session: AsyncSession, conversation_id: str
) -> PendingClarification | None:
    pending = await session.get(PendingClarification, conversation_id)

    if pending is None:
        return None

    if datetime.utcnow() - pending.created_at > timedelta(hours=TTL_HOURS):
        await clear_pending_clarification(session, conversation_id)
        return None

    return pending


async def clear_pending_clarification(
    session: AsyncSession, conversation_id: str
) -> None:
    await session.execute(
        delete(PendingClarification).where(
            PendingClarification.conversation_id == conversation_id
        )
    )
    await session.commit()