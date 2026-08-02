import logging

from app.models.request import EscalationReason, RequestResult, RequestType
from app.repositories.pending_clarification_repo import (
    get_pending_clarification,
    clear_pending_clarification,
)
from app.repositories.request_repository import RequestRepository
from app.services.escalation import escalate_to_operator
from app.services.knowledge_base import KnowledgeBase
from app.services.intercom_client import IntercomClient

logger = logging.getLogger("clarification_handler")


def normalize_answer(text: str) -> str:
    return text.strip().lower()


async def handle_clarification_response(
    session,
    conversation_id: str,
    answer: str,
    knowledge_base: KnowledgeBase,
    intercom_client: IntercomClient,
    is_ticket: bool,
    repository: RequestRepository,
    request_type: RequestType,
    user_name: str = "",
    user_email: str = "",
    updated_at: str = "",
) -> bool:
    pending = await get_pending_clarification(session, conversation_id)

    if pending is None:
        return False

    normalized = normalize_answer(answer)

    wants_new_school = any(
        phrase in normalized for phrase in ["в новую", "новую школу", "перейти", "сменить"]
    )
    wants_to_leave_only = any(
        phrase in normalized for phrase in ["выйти", "покинуть", "уйти"]
    ) and not wants_new_school

    if wants_new_school:
        logger.info("Clarification resolved: wants new school | id=%s", conversation_id)

        await escalate_to_operator(
            conversation_id=conversation_id,
            question=answer,
            reason=EscalationReason.WANTS_NEW_SCHOOL.value,
            is_ticket=is_ticket,
            intercom_client=intercom_client,
            user_name=user_name,
            user_email=user_email,
            updated_at=updated_at,
        )

        await repository.create(
            conversation_id=conversation_id,
            request_type=request_type,
            result=RequestResult.ESCALATED,
            escalation_reason=EscalationReason.WANTS_NEW_SCHOOL,
            question=answer,
        )

        await clear_pending_clarification(session, conversation_id)
        return True

    if wants_to_leave_only:
        logger.info("Clarification resolved: leave only | id=%s", conversation_id)

        category = knowledge_base.get_by_id(pending.category_id)

        if category:
            template = category["template_answer"]
            if is_ticket:
                await intercom_client.send_ticket_reply(conversation_id, template)
            else:
                await intercom_client.send_conversations_reply(conversation_id, template)

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.TEMPLATE,
                category=category["category"],
                question=answer,
                answer=template,
            )
        else:
            await escalate_to_operator(
                conversation_id=conversation_id,
                question=answer,
                reason=EscalationReason.NO_CATEGORY.value,
                is_ticket=is_ticket,
                intercom_client=intercom_client,
                user_name=user_name,
                user_email=user_email,
                updated_at=updated_at,
            )

        await clear_pending_clarification(session, conversation_id)
        return True

    # ответ неоднозначный — переспрашиваем ещё раз
    logger.info(
        "Clarification unclear, re-asking | id=%s | answer=%s",
        conversation_id, answer[:80]
    )

    reply = (
        "Пожалуйста, уточните одним из вариантов:\n"
        "«Выйти из школы» — если хотите просто уйти\n"
        "«В новую» — если хотите перейти в другую школу"
    )

    if is_ticket:
        await intercom_client.send_ticket_reply(conversation_id, reply)
    else:
        await intercom_client.send_conversations_reply(conversation_id, reply)

    return True