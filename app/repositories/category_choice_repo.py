from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_category_choice import (
    CategoryChoiceStatus,
    ConversationCategoryChoice,
)


async def get_pending_category_choice(
    session: AsyncSession,
    conversation_id: str,
) -> ConversationCategoryChoice | None:
    result = await session.execute(
        select(ConversationCategoryChoice).where(
            ConversationCategoryChoice.conversation_id == conversation_id,
            ConversationCategoryChoice.status == CategoryChoiceStatus.PENDING,
        )
    )
    return result.scalar_one_or_none()


async def create_category_choice(
    session: AsyncSession,
    *,
    conversation_id: str,
    question: str,
    candidate_ids: list[str],
) -> ConversationCategoryChoice:
    if len(candidate_ids) < 2 or len(candidate_ids) > 3:
        raise ValueError("Category choice requires 2 or 3 candidates")

    # Ищем любую существующую запись по conversation_id
    result = await session.execute(
        select(ConversationCategoryChoice).where(
            ConversationCategoryChoice.conversation_id == conversation_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.question = question
        existing.candidate_1 = candidate_ids[0]
        existing.candidate_2 = candidate_ids[1]
        existing.candidate_3 = candidate_ids[2] if len(candidate_ids) == 3 else None
        existing.status = CategoryChoiceStatus.PENDING
        existing.attempts = 0
        existing.selected_candidates = None
        return existing

    state = ConversationCategoryChoice(
        conversation_id=conversation_id,
        question=question,
        candidate_1=candidate_ids[0],
        candidate_2=candidate_ids[1],
        candidate_3=candidate_ids[2] if len(candidate_ids) == 3 else None,
        status=CategoryChoiceStatus.PENDING,
        attempts=0,
    )
    session.add(state)
    await session.flush()
    return statew


async def increment_attempts(
    session: AsyncSession,
    state: ConversationCategoryChoice,
) -> int:
    state.attempts += 1
    await session.flush()
    return state.attempts


async def resolve_category_choice(
    session: AsyncSession,
    state: ConversationCategoryChoice,
    selected_ids: list[str],
) -> None:
    state.status = CategoryChoiceStatus.RESOLVED
    state.selected_candidates = ",".join(selected_ids)
    await session.flush()


async def cancel_category_choice(
    session: AsyncSession,
    state: ConversationCategoryChoice,
) -> None:
    state.status = CategoryChoiceStatus.CANCELLED
    await session.flush()
