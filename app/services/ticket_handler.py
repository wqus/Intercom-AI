import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import (
    EscalationReason,
    RequestResult,
    RequestType,
)
from app.repositories.request_repository import RequestRepository
from app.repositories.pending_clarification_repo import save_pending_clarification
from app.services.escalation import escalate_to_operator
from app.services.intercom_client import IntercomClient
from app.services.knowledge_base import KnowledgeBase, MatchMode
from app.services.llm_client import YandexLLMClient
from app.services.clarification_handler import handle_clarification_response
from app.services.message_filter import is_non_query_message

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
        # === ШАГ 0: проверка pending clarification ===
        was_clarification = await handle_clarification_response(
            session=session,
            conversation_id=conversation_id,
            answer=question,
            knowledge_base=knowledge_base,
            intercom_client=intercom_client,
            is_ticket=is_ticket,
            repository=repository,
            request_type=request_type,
            user_name=user_name,
            user_email=user_email,
            updated_at=updated_at,
        )
        if was_clarification:
            return

        # === ШАГ 0.5: фильтр не-запросов ===
        if is_non_query_message(question):
            logger.info(
                "Non-query message, ignoring | id=%s | text=%s",
                conversation_id, question[:60]
            )

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.IGNORED,
                category=None,
                question=question,
                answer=None,
            )

            return

        #Fuzzy match

        match = knowledge_base.match(question)

        logger.info(
            "KB match | mode=%s | score=%.1f | category=%s",
            match.mode,
            match.score,
            match.category,
        )

        #CLARIFICATION NEEDED (fuzzy match путь)

        if match.mode == MatchMode.CLARIFICATION:
            logger.info(
                "Category requires clarification | category=%s",
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

            await save_pending_clarification(
                session=session,
                conversation_id=conversation_id,
                category_id=match.kb_id,
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
            "LLM result | is_noise=%s | can_answer=%s | needs_clarification=%s",
            llm_result.is_noise,
            llm_result.can_answer,
            llm_result.needs_clarification,
        )

        #LLM NOISE

        if llm_result.is_noise:
            logger.info("LLM classified as noise | id=%s", conversation_id)

            await repository.create(
                conversation_id=conversation_id,
                request_type=request_type,
                result=RequestResult.IGNORED,
                category=match.category,
                question=question,
                answer=None,
            )

            return

        #LLM CLARIFICATION (новый путь через LLM)

        if llm_result.needs_clarification and llm_result.clarification_kb_id:
            category = knowledge_base.get_by_id(llm_result.clarification_kb_id)

            if category and category.get("clarification_question"):
                logger.info(
                    "LLM triggered clarification | kb_id=%s",
                    llm_result.clarification_kb_id,
                )

                if is_ticket:
                    await intercom_client.send_ticket_reply(
                        conversation_id, category["clarification_question"]
                    )
                else:
                    await intercom_client.send_conversations_reply(
                        conversation_id, category["clarification_question"]
                    )

                await save_pending_clarification(
                    session=session,
                    conversation_id=conversation_id,
                    category_id=llm_result.clarification_kb_id,
                )

                await repository.create(
                    conversation_id=conversation_id,
                    request_type=request_type,
                    result=RequestResult.TEMPLATE,
                    category=category["category"],
                    question=question,
                    answer=category["clarification_question"],
                )

                return

            logger.warning(
                "LLM requested clarification for unknown/invalid kb_id=%s — falling back to escalation",
                llm_result.clarification_kb_id,
            )
            # если kb_id некорректный — не теряем тикет, эскалируем как обычно
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

            return

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