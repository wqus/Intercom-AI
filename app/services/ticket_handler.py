import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import (
    EscalationReason,
    RequestResult,
    RequestType,
)
from app.repositories.request_repository import RequestRepository
from app.services.escalation import escalate_to_operator
from app.services.intercom_client import IntercomClient
from app.services.knowledge_base import KnowledgeBase, MatchMode
from app.services.llm_client import YandexLLMClient

logger = logging.getLogger("ticket_handler")


async def handle_new_ticket(
    conversation_id: str,
    question: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
    session: AsyncSession,
    is_ticket: bool = False,
    intercom_client: IntercomClient | None = None,
    user_name: str = "",
    user_email: str = "",
    updated_at: str = ""
) -> None:

    if intercom_client is None:
        intercom_client = IntercomClient()

    repository = RequestRepository(session)

    request_type = (
        RequestType.TICKET
        if is_ticket
        else RequestType.CONVERSATION
    )

    logger.info(
        "Processing request | id=%s | question=%s",
        conversation_id,
        question[:80],
    )

    try:
        #Fuzzy match

        match = knowledge_base.match(question)

        logger.info(
            "KB match | mode=%s | score=%.1f | category=%s",
            match.mode,
            match.score,
            match.category,
        )

        #TEMPLATE

        if match.mode == MatchMode.TEMPLATE:
            logger.info(
                "Sending template answer | category=%s",
                match.category,
            )

            if is_ticket:
                await intercom_client.send_ticket_reply(
                    conversation_id,
                    match.template_answer,
                )
            else:
                await intercom_client.send_conversations_reply(
                    conversation_id,
                    match.template_answer,
                )

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.TEMPLATE,
                category=match.category,
                question=question,
                answer=match.template_answer,
            )

            return

        #USER REQUESTED OPERATOR

        if match.mode == MatchMode.OPERATOR:
            logger.info("User requested an operator")

            await escalate_to_operator(
                conversation_id=conversation_id,
                question=question,
                reason=EscalationReason.USER_REQUESTED_OPERATOR.value,
                is_ticket=is_ticket,
                intercom_client=intercom_client,
                user_name=user_name,
                user_email=user_email,
                updated_at=updated_at
            )

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.ESCALATED,
                escalation_reason=EscalationReason.USER_REQUESTED_OPERATOR,
                question=question,
            )

            return

        #NO MATCH

        if match.mode == MatchMode.NO_MATCH or not match.relevant_entries:
            logger.info("No KB match")

            await escalate_to_operator(
                conversation_id=conversation_id,
                question=question,
                reason=EscalationReason.NO_CATEGORY.value,
                is_ticket=is_ticket,
                intercom_client=intercom_client,
                user_name=user_name,
                user_email=user_email,
                updated_at=updated_at
            )

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.ESCALATED,
                escalation_reason=EscalationReason.NO_CATEGORY,
                question=question,
            )

            return

        #LLM

        logger.info(
            "Sending to LLM | relevant_entries=%d | category=%s",
            len(match.relevant_entries),
            match.category,
        )

        llm_result = await llm_client.classify(
            question=question,
            relevant_entries=match.relevant_entries,
        )

        logger.info(
            "LLM result | can_answer=%s",
            llm_result.can_answer,
        )

        #LLM ANSWER

        if llm_result.can_answer and llm_result.answer:
            if is_ticket:
                await intercom_client.send_ticket_reply(
                    conversation_id,
                    llm_result.answer,
                )
            else:
                await intercom_client.send_conversations_reply(
                    conversation_id,
                    llm_result.answer,
                )

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.LLM,
                category=match.category,
                question=question,
                answer=llm_result.answer,
            )

            return

        #LLM CANNOT ANSWER

        logger.info(
            "LLM cannot answer | reason=%s",
            llm_result.reason,
        )

        await escalate_to_operator(
            conversation_id=conversation_id,
            question=question,
            reason=EscalationReason.LLM_CANNOT_ANSWER.value,
            is_ticket=is_ticket,
            intercom_client=intercom_client,
            user_name=user_name,
            user_email=user_email,
            updated_at=updated_at
        )

        await repository.create(
            conversation_id=conversation_id,
            request_type=request_type,
            result=RequestResult.ESCALATED,
            escalation_reason=EscalationReason.LLM_CANNOT_ANSWER,
            category=match.category,
            question=question,
        )

    except Exception as e:
        logger.exception(
            "Unexpected error while processing request %s",
            conversation_id,
        )

        try:
            await escalate_to_operator(
                conversation_id=conversation_id,
                question=question,
                reason=EscalationReason.INTERNAL_ERROR.value,
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
                escalation_reason=EscalationReason.INTERNAL_ERROR,
                question=question,
            )

        except Exception:
            logger.exception(
                "Failed to escalate request %s",
                conversation_id,
            )

    finally:
        await intercom_client.close()