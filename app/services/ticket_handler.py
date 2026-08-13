from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import (
    EscalationReason,
    RequestResult,
    RequestType,
)
from app.repositories.pending_clarification_repo import save_pending_clarification
from app.repositories.request_repository import RequestRepository
from app.services.clarification_handler import handle_clarification_response
from app.services.escalation import escalate_to_operator
from app.services.intercom_client import IntercomClient
from app.services.knowledge_base import (
    KnowledgeBase,
    MatchMode,
    MatchResult,
    contains_operator_request,
)
from app.services.llm_client import YandexLLMClient
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
    updated_at: str = "",
) -> None:
    """Execute the KB -> template/clarification/LLM -> operator pipeline."""

    owns_intercom_client = intercom_client is None
    client = intercom_client or IntercomClient()
    repository = RequestRepository(session)
    request_type = RequestType.TICKET if is_ticket else RequestType.CONVERSATION
    question = (question or "").strip()

    async def send_reply(answer: str) -> None:
        if is_ticket:
            await client.send_ticket_reply(conversation_id, answer)
        else:
            await client.send_conversations_reply(conversation_id, answer)

    async def save_result(
        result: RequestResult,
        *,
        category: str | None = None,
        answer: str | None = None,
        escalation_reason: EscalationReason | None = None,
        question_text: str | None = None,
    ) -> None:
        await repository.create(
            conversation_id=conversation_id,
            request_type=request_type,
            result=result,
            category=category,
            question=question if question_text is None else question_text,
            answer=answer,
            escalation_reason=escalation_reason,
        )

    async def escalate(
        reason: EscalationReason,
        *,
        category: str | None = None,
        question_text: str | None = None,
    ) -> None:
        await escalate_to_operator(
            conversation_id=conversation_id,
            question=question if question_text is None else question_text,
            reason=reason.value,
            is_ticket=is_ticket,
            intercom_client=client,
            user_name=user_name,
            user_email=user_email,
            updated_at=updated_at,
        )
        await save_result(
            RequestResult.ESCALATED,
            category=category,
            escalation_reason=reason,
            question_text=question_text,
        )

    try:
        logger.info(
            "Processing request | id=%s | question=%s",
            conversation_id,
            question[:120],
        )

        if not question:
            await send_reply(knowledge_base.default_answer)
            await save_result(RequestResult.TEMPLATE, answer=knowledge_base.default_answer)
            return

        # Direct operator request always wins, even over an existing clarification state.
        if contains_operator_request(question):
            await escalate(EscalationReason.USER_REQUESTED_OPERATOR)
            return

        # If the user is answering a pending clarification, consume it before
        # normal intent classification.
        if await handle_clarification_response(
            session=session,
            conversation_id=conversation_id,
            answer=question,
            knowledge_base=knowledge_base,
            intercom_client=client,
            is_ticket=is_ticket,
            repository=repository,
            request_type=request_type,
            user_name=user_name,
            user_email=user_email,
            updated_at=updated_at,
        ):
            return

        if is_non_query_message(question):
            logger.info("Non-query ignored | id=%s", conversation_id)
            await save_result(RequestResult.IGNORED)
            return

        matches = knowledge_base.match_many(question)

        if len(matches) == 1:
            await _execute_match(
                match=matches[0],
                original_question=question,
                conversation_id=conversation_id,
                llm_client=llm_client,
                knowledge_base=knowledge_base,
                session=session,
                repository=repository,
                request_type=request_type,
                is_ticket=is_ticket,
                client=client,
                send_reply=send_reply,
                save_result=save_result,
                escalate=escalate,
            )
            return

        await _execute_multiple_matches(
            matches=matches,
            original_question=question,
            conversation_id=conversation_id,
            llm_client=llm_client,
            knowledge_base=knowledge_base,
            session=session,
            client=client,
            send_reply=send_reply,
            save_result=save_result,
            escalate=escalate,
        )

    except Exception:
        logger.exception("Unexpected error while processing request %s", conversation_id)
        try:
            await escalate(EscalationReason.INTERNAL_ERROR)
        except Exception:
            logger.exception("Failed to escalate request %s", conversation_id)
    finally:
        if owns_intercom_client:
            await client.close()


async def _execute_match(
    *,
    match: MatchResult,
    original_question: str,
    conversation_id: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
    session: AsyncSession,
    repository: RequestRepository,
    request_type: RequestType,
    is_ticket: bool,
    client: IntercomClient,
    send_reply: Callable[[str], Awaitable[None]],
    save_result: Callable[..., Awaitable[None]],
    escalate: Callable[..., Awaitable[None]],
) -> None:
    logger.info(
        "KB decision | id=%s | mode=%s | kb_id=%s | score=%.4f | margin=%.4f | embedding=%.4f",
        conversation_id,
        match.mode.value,
        match.kb_id,
        match.score,
        match.margin,
        match.embedding_score,
    )

    if match.mode == MatchMode.OPERATOR:
        await escalate(
            EscalationReason.USER_REQUESTED_OPERATOR,
            category=match.category,
            question_text=original_question,
        )
        return

    if match.mode == MatchMode.NO_MATCH:
        await escalate(
            EscalationReason.NO_CATEGORY,
            category=match.category,
            question_text=original_question,
        )
        return

    if match.mode == MatchMode.CLARIFICATION:
        if not match.kb_id or not match.template_answer:
            logger.error("Invalid clarification result | kb_id=%s", match.kb_id)
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                category=match.category,
                question_text=original_question,
            )
            return

        await send_reply(match.template_answer)
        await save_pending_clarification(
            session=session,
            conversation_id=conversation_id,
            category_id=match.kb_id,
        )
        await save_result(
            RequestResult.TEMPLATE,
            category=match.category,
            answer=match.template_answer,
            question_text=original_question,
        )
        return

    if match.mode == MatchMode.TEMPLATE:
        if not match.template_answer:
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                category=match.category,
                question_text=original_question,
            )
            return

        await send_reply(match.template_answer)
        await save_result(
            RequestResult.TEMPLATE,
            category=match.category,
            answer=match.template_answer,
            question_text=original_question,
        )
        return

    if match.mode == MatchMode.LLM:
        await _execute_llm(
            match=match,
            question=match.subquestion or original_question,
            conversation_id=conversation_id,
            llm_client=llm_client,
            knowledge_base=knowledge_base,
            session=session,
            send_reply=send_reply,
            save_result=save_result,
            escalate=escalate,
        )
        return

    logger.error("Unhandled MatchMode=%s", match.mode)
    await escalate(
        EscalationReason.INTERNAL_ERROR,
        category=match.category,
        question_text=original_question,
    )


async def _execute_llm(
    *,
    match: MatchResult,
    question: str,
    conversation_id: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
    session: AsyncSession,
    send_reply: Callable[[str], Awaitable[None]],
    save_result: Callable[..., Awaitable[None]],
    escalate: Callable[..., Awaitable[None]],
) -> None:
    candidates = match.relevant_entries
    candidate_ids = {
        str(entry.get("id"))
        for entry in candidates
        if entry.get("id")
    }

    if not candidate_ids:
        await escalate(EscalationReason.NO_CATEGORY, category=match.category, question_text=question)
        return

    logger.info(
        "Calling constrained LLM | id=%s | candidates=%s | top=%s",
        conversation_id,
        sorted(candidate_ids),
        match.kb_id,
    )

    result = await llm_client.classify(
        question=question,
        relevant_entries=candidates,
    )

    if result.is_noise:
        await save_result(
            RequestResult.IGNORED,
            category=match.category,
            question_text=question,
        )
        return

    if result.needs_clarification:
        kb_id = str(result.clarification_kb_id or "")

        # The LLM may only select a category already retrieved by our KB.
        if kb_id not in candidate_ids:
            logger.warning(
                "LLM returned out-of-candidate clarification | kb_id=%s | candidates=%s",
                kb_id,
                sorted(candidate_ids),
            )
            await escalate(
                EscalationReason.LLM_CANNOT_ANSWER,
                category=match.category,
                question_text=question,
            )
            return

        category = knowledge_base.get_by_id(kb_id)
        clarification = category.get("clarification_question") if category else None
        if not clarification:
            await escalate(
                EscalationReason.LLM_CANNOT_ANSWER,
                category=match.category,
                question_text=question,
            )
            return

        await send_reply(clarification)
        await save_pending_clarification(
            session=session,
            conversation_id=conversation_id,
            category_id=kb_id,
        )
        await save_result(
            RequestResult.TEMPLATE,
            category=category.get("category") if category else match.category,
            answer=clarification,
            question_text=question,
        )
        return

    if result.can_answer and result.answer:
        await send_reply(result.answer)
        await save_result(
            RequestResult.LLM,
            category=match.category,
            answer=result.answer,
            question_text=question,
        )
        return

    await escalate(
        EscalationReason.LLM_CANNOT_ANSWER,
        category=match.category,
        question_text=question,
    )


async def _execute_multiple_matches(
    *,
    matches: list[MatchResult],
    original_question: str,
    conversation_id: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
    session: AsyncSession,
    client: IntercomClient,
    send_reply: Callable[[str], Awaitable[None]],
    save_result: Callable[..., Awaitable[None]],
    escalate: Callable[..., Awaitable[None]],
) -> None:
    """Answer multiple independent intents without silently dropping an unknown one."""

    deterministic_answers: list[str] = []
    ambiguous: list[MatchResult] = []
    clarifications: list[MatchResult] = []
    no_matches: list[MatchResult] = []
    operators: list[MatchResult] = []
    seen_categories: set[str] = set()

    for match in matches:
        if match.mode == MatchMode.TEMPLATE and match.template_answer:
            key = match.kb_id or match.category or "template"
            if key not in seen_categories:
                deterministic_answers.append(match.template_answer)
                seen_categories.add(key)
        elif match.mode == MatchMode.LLM:
            ambiguous.append(match)
        elif match.mode == MatchMode.CLARIFICATION:
            clarifications.append(match)
        elif match.mode == MatchMode.NO_MATCH:
            no_matches.append(match)
        elif match.mode == MatchMode.OPERATOR:
            operators.append(match)

    if operators:
        await escalate(
            EscalationReason.USER_REQUESTED_OPERATOR,
            category=operators[0].category,
            question_text=original_question,
        )
        return

    # A clarification changes the state of the conversation. Do not combine it
    # with an ambiguous LLM conversation state.
    if clarifications:
        if len(clarifications) != 1 or ambiguous or no_matches:
            await escalate(
                EscalationReason.LLM_CANNOT_ANSWER,
                category=clarifications[0].category,
                question_text=original_question,
            )
            return

        clarification = clarifications[0]
        if not clarification.kb_id or not clarification.template_answer:
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                category=clarification.category,
                question_text=original_question,
            )
            return

        response_parts = deterministic_answers + [clarification.template_answer]
        response = "\n\n".join(response_parts)
        await send_reply(response)
        await save_pending_clarification(
            session=session,
            conversation_id=conversation_id,
            category_id=clarification.kb_id,
        )
        await save_result(
            RequestResult.TEMPLATE,
            category="multiple",
            answer=response,
            question_text=original_question,
        )
        return

    # Unknown part means we cannot honestly claim the whole message was answered.
    if no_matches:
        await escalate(
            EscalationReason.NO_CATEGORY,
            category=no_matches[0].category,
            question_text=original_question,
        )
        return

    if not ambiguous:
        if deterministic_answers:
            response = "\n\n".join(deterministic_answers)
            await send_reply(response)
            await save_result(
                RequestResult.TEMPLATE,
                category="multiple",
                answer=response,
                question_text=original_question,
            )
            return

        await escalate(EscalationReason.NO_CATEGORY, question_text=original_question)
        return

    # Resolve each ambiguous part independently with the constrained LLM.
    answers = list(deterministic_answers)

    for match in ambiguous:
        candidates = match.relevant_entries
        candidate_ids = {str(entry.get("id")) for entry in candidates if entry.get("id")}
        if not candidate_ids:
            await escalate(EscalationReason.LLM_CANNOT_ANSWER, category=match.category, question_text=original_question)
            return

        subquestion = match.subquestion or original_question
        result = await llm_client.classify(
            question=subquestion,
            relevant_entries=candidates,
        )

        if result.is_noise:
            continue

        if result.needs_clarification:
            kb_id = str(result.clarification_kb_id or "")
            if kb_id not in candidate_ids:
                await escalate(EscalationReason.LLM_CANNOT_ANSWER, category=match.category, question_text=original_question)
                return

            category = knowledge_base.get_by_id(kb_id)
            clarification = category.get("clarification_question") if category else None
            if not clarification:
                await escalate(EscalationReason.LLM_CANNOT_ANSWER, category=match.category, question_text=original_question)
                return

            # More than one state transition in one message is deliberately conservative.
            await escalate(EscalationReason.LLM_CANNOT_ANSWER, category=match.category, question_text=original_question)
            return

        if result.can_answer and result.answer:
            answers.append(result.answer)
        else:
            await escalate(EscalationReason.LLM_CANNOT_ANSWER, category=match.category, question_text=original_question)
            return

    if not answers:
        await escalate(EscalationReason.LLM_CANNOT_ANSWER, question_text=original_question)
        return

    response = "\n\n".join(answers)
    await send_reply(response)
    await save_result(
        RequestResult.LLM,
        category="multiple",
        answer=response,
        question_text=original_question,
    )