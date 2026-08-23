from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_rephrase_state import ConversationRephraseState


async def get_rephrase_state(
    session: AsyncSession,
    conversation_id: str,
) -> ConversationRephraseState | None:
    result = await session.execute(
        select(ConversationRephraseState).where(
            ConversationRephraseState.conversation_id == conversation_id
        )
    )
    return result.scalar_one_or_none()


async def register_failure(
    session: AsyncSession,
    conversation_id: str,
    original_question: str,
) -> ConversationRephraseState:
    state = await get_rephrase_state(session, conversation_id)

    if state is None:
        state = ConversationRephraseState(
            conversation_id=conversation_id,
            attempts=1,
            original_question=original_question,
        )
        session.add(state)
    else:
        state.attempts += 1
        state.original_question = original_question

    await session.flush()
    return state


async def clear_rephrase_state(
    session: AsyncSession,
    conversation_id: str,
) -> None:
    state = await get_rephrase_state(session, conversation_id)
    if state is not None:
        await session.delete(state)
        await session.flush()
