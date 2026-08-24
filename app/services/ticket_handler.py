
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import EscalationReason, RequestResult, RequestType
from app.repositories.pending_clarification_repo import save_pending_clarification
from app.repositories.rephrase_state_repo import get_rephrase_state, register_failure, clear_rephrase_state
from app.repositories.category_choice_repo import (
    get_pending_category_choice,
    create_category_choice,
    increment_attempts as increment_choice_attempts,
    resolve_category_choice,
    cancel_category_choice,
)
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
from app.services.conversation_state_orchestrator import (
    MAX_REPHRASE_FAILURES,
    MAX_CATEGORY_CHOICE_ATTEMPTS,
    candidate_ids_from_state,
    selected_candidate_ids_from_text,
)

logger = logging.getLogger("ticket_handler")


async def _send_rephrase_prompt(
    send_reply: Callable[[str], Awaitable[None]],
) -> None:
    await send_reply(
        "Не удалось точно понять ваш вопрос. "
        "Пожалуйста, переформулируйте его немного подробнее."
    )


async def _start_or_continue_rephrase(
    *,
    session: AsyncSession,
    conversation_id: str,
    question: str,
    send_reply: Callable[[str], Awaitable[None]],
    escalate: Callable[..., Awaitable[None]],
) -> bool:
    """Return True when the request is fully handled by the REPHRASE flow."""
    state = await register_failure(
        session=session,
        conversation_id=conversation_id,
        original_question=question,
    )

    if state.attempts >= MAX_REPHRASE_FAILURES:
        await clear_rephrase_state(session, conversation_id)
        await escalate(
            EscalationReason.NO_CATEGORY,
            question_text=question,
        )
        return True

    await _send_rephrase_prompt(send_reply)
    return True


async def _handle_pending_category_choice(
    *,
    session: AsyncSession,
    conversation_id: str,
    question: str,
    knowledge_base: KnowledgeBase,
    send_reply: Callable[[str], Awaitable[None]],
    llm_client: YandexLLMClient,
    repository: RequestRepository,
    save_result: Callable[..., Awaitable[None]],
    escalate: Callable[..., Awaitable[None]],
) -> bool:
    """Consume an active CATEGORY_CHOICE state before ordinary classification."""
    state = await get_pending_category_choice(session, conversation_id)
    if state is None:
        return False

    candidate_ids = candidate_ids_from_state(state)
    selected_ids = selected_candidate_ids_from_text(
        knowledge_base,
        state,
        question,
    )

    if not selected_ids:
        attempts = await increment_choice_attempts(session, state)
        if attempts >= MAX_CATEGORY_CHOICE_ATTEMPTS:
            await cancel_category_choice(session, state)
            await escalate(
                EscalationReason.NO_CATEGORY,
                question_text=question,
            )
            return True

        await send_reply(
            f"Пожалуйста, выберите номер одного или нескольких вариантов: "
            f"{', '.join(str(i) for i in range(1, len(candidate_ids) + 1))}."
            f"Если ни один из предложенных варинтов не подходит, напишите \"оператор\" для передачи вашего запроса специалисту"

        )
        return True

    await resolve_category_choice(session, state, selected_ids)

    answers: list[str] = []
    categories: list[str] = []

    for kb_id in selected_ids:
        entry = knowledge_base.get_by_id(kb_id)
        if entry is None:
            logger.error(
                "CATEGORY_CHOICE references missing KB id | conv=%s | kb_id=%s",
                conversation_id,
                kb_id,
            )
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                question_text=state.question,
            )
            return True

        if entry.get("requires_clarification"):
            # Existing stateful clarification is one-at-a-time.
            if len(selected_ids) != 1:
                await escalate(
                    EscalationReason.LLM_CANNOT_ANSWER,
                    category=entry.get("category"),
                    question_text=state.question,
                )
                return True

            clarification = entry.get("clarification_question")
            if not clarification:
                await escalate(
                    EscalationReason.INTERNAL_ERROR,
                    category=entry.get("category"),
                    question_text=state.question,
                )
                return True

            await send_reply(clarification)
            await save_pending_clarification(
                session=session,
                conversation_id=conversation_id,
                category_id=kb_id,
            )
            await save_result(
                RequestResult.TEMPLATE,
                category=entry.get("category"),
                answer=clarification,
                question_text=state.question,
            )
            return True

        answer = entry.get("template_answer")
        if not answer:
            # A selected category should still be answerable. Give the
            # constrained LLM a single-category decision context.
            result = await llm_client.classify(
                question=state.question,
                relevant_entries=[knowledge_base._clean_entry_for_llm(entry)],
            )
            if not result.can_answer or not result.answer:
                await escalate(
                    EscalationReason.LLM_CANNOT_ANSWER,
                    category=entry.get("category"),
                    question_text=state.question,
                )
                return True
            answer = result.answer

        answers.append(answer)
        if entry.get("category"):
            categories.append(str(entry["category"]))

    response = "\n\n".join(answers)
    await send_reply(response)
    await save_result(
        RequestResult.TEMPLATE,
        category="multiple" if len(categories) > 1 else (categories[0] if categories else None),
        answer=response,
        question_text=state.question,
    )
    return True


async def _maybe_start_category_choice(
    *,
    session: AsyncSession,
    conversation_id: str,
    question: str,
    knowledge_base: KnowledgeBase,
    send_reply: Callable[[str], Awaitable[None]],
    save_result: Callable[..., Awaitable[None]],
) -> bool:
    candidates = knowledge_base.get_category_choice_candidates(question)
    if len(candidates) < 2:
        return False

    candidate_ids = [str(candidate["kb_id"]) for candidate in candidates]
    await create_category_choice(
        session=session,
        conversation_id=conversation_id,
        question=question,
        candidate_ids=candidate_ids,
    )

    await send_reply(
        knowledge_base.format_category_choice(candidates)
    )
    await save_result(
        RequestResult.TEMPLATE,
        category="category_choice",
        answer=knowledge_base.format_category_choice(candidates),
        question_text=question,
    )
    return True


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
    """KB -> template/clarification/LLM -> operator pipeline.

    Each sub-question produced by ``match_many`` is handled independently.
    Existing intent-specific clarification remains one-at-a-time.
    """
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
        match: MatchResult | None = None,
    ) -> None:
        await repository.create(
            conversation_id=conversation_id,
            request_type=request_type,
            result=result,
            category=category,
            question=question if question_text is None else question_text,
            answer=answer,
            escalation_reason=escalation_reason,
            predicted_kb_id=match.kb_id if match else None,
            subquestion=match.subquestion if match else None,
            score=match.score if match else None,
            second_score=match.second_score if match else None,
            margin=match.margin if match else None,
            embedding_score=match.embedding_score if match else None,
            intent_fit_score=match.intent_fit_score if match else None,
            decision_reason=(match.decision_reason[:100] if match and match.decision_reason else None),
        )

    async def escalate(
        reason: EscalationReason,
        *,
        category: str | None = None,
        question_text: str | None = None,
        match: MatchResult | None = None,
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
            match=match,
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

        if contains_operator_request(question):
            await escalate(EscalationReason.USER_REQUESTED_OPERATOR)
            return

        # A pending CATEGORY_CHOICE is a strict conversational state:
        # numeric selection is expected instead of ordinary classification.
        if await _handle_pending_category_choice(
            session=session,
            conversation_id=conversation_id,
            question=question,
            knowledge_base=knowledge_base,
            send_reply=send_reply,
            llm_client=llm_client,
            repository=repository,
            save_result=save_result,
            escalate=escalate,
        ):
            return

        # Existing intent-specific clarification remains a separate state.
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
        logger.info(
            "KB match_many | id=%s | parts=%d | modes=%s | subquestions=%s",
            conversation_id,
            len(matches),
            [m.mode.value for m in matches],
            [m.subquestion for m in matches],
        )

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
    question_for_result = match.subquestion or original_question

    logger.info(
        "KB decision | id=%s | mode=%s | kb_id=%s | score=%.4f | margin=%.4f | embedding=%.4f | reason=%s | subquestion=%s",
        conversation_id,
        match.mode.value,
        match.kb_id,
        match.score,
        match.margin,
        match.embedding_score,
        match.decision_reason,
        match.subquestion,
    )

    if match.mode == MatchMode.OPERATOR:
        await escalate(
            EscalationReason.USER_REQUESTED_OPERATOR,
            category=match.category,
            question_text=question_for_result,
            match=match,
        )
        return

    if match.mode == MatchMode.NO_MATCH:
        if await _maybe_start_category_choice(
            session=session,
            conversation_id=conversation_id,
            question=question_for_result,
            knowledge_base=knowledge_base,
            send_reply=send_reply,
            save_result=save_result,
        ):
            return

        if knowledge_base.is_rephrase_candidate(match):
            await _start_or_continue_rephrase(
                session=session,
                conversation_id=conversation_id,
                question=question_for_result,
                send_reply=send_reply,
                escalate=escalate,
            )
            return

        await escalate(
            EscalationReason.NO_CATEGORY,
            category=match.category,
            question_text=question_for_result,
            match=match,
        )
        return

    if match.mode == MatchMode.CLARIFICATION:
        if not match.kb_id or not match.template_answer:
            logger.error("Invalid clarification result | kb_id=%s", match.kb_id)
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                category=match.category,
                question_text=question_for_result,
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
            question_text=question_for_result,
            match=match,
        )
        return

    if match.mode == MatchMode.TEMPLATE:
        if not match.template_answer:
            await escalate(
                EscalationReason.INTERNAL_ERROR,
                category=match.category,
                question_text=question_for_result,
            )
            return

        await send_reply(match.template_answer)
        await save_result(
            RequestResult.TEMPLATE,
            category=match.category,
            answer=match.template_answer,
            question_text=question_for_result,
            match=match,
        )
        return

    if match.mode == MatchMode.LLM:
        await _execute_llm(
            match=match,
            question=question_for_result,
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
        question_text=question_for_result,
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
        await escalate(
            EscalationReason.NO_CATEGORY,
            category=match.category,
            question_text=question,
        )
        return

    logger.info(
        "Calling constrained LLM | id=%s | candidates=%s | top=%s | subquestion=%r",
        conversation_id,
        sorted(candidate_ids),
        match.kb_id,
        question,
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

    # The LLM must select one of the retrieved candidates when it answers.
    if result.can_answer:
        selected_kb_id = str(result.selected_kb_id or "")
        if selected_kb_id not in candidate_ids:
            logger.warning(
                "LLM returned invalid selected_kb_id | selected=%s | candidates=%s",
                selected_kb_id,
                sorted(candidate_ids),
            )
            await escalate(
                EscalationReason.LLM_CANNOT_ANSWER,
                category=match.category,
                question_text=question,
            )
            return

        if not result.answer:
            await escalate(
                EscalationReason.LLM_CANNOT_ANSWER,
                category=match.category,
                question_text=question,
            )
            return

        selected_entry = knowledge_base.get_by_id(selected_kb_id)
        selected_category = (
            selected_entry.get("category")
            if selected_entry
            else match.category
        )

        await send_reply(result.answer)
        await save_result(
            RequestResult.LLM,
            category=selected_category,
            answer=result.answer,
            question_text=question,
            match=match,
        )
        return

    if result.needs_clarification:
        kb_id = str(result.clarification_kb_id or "")

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

        # Existing intent-specific clarification remains one-at-a-time.
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
            match=match,
        )
        return

    # LLM could not safely answer. Give the user a choice if retrieval
    # still found multiple sufficiently strong candidates.
    if await _maybe_start_category_choice(
        session=session,
        conversation_id=conversation_id,
        question=question,
        knowledge_base=knowledge_base,
        send_reply=send_reply,
        save_result=save_result,
    ):
        return

    if knowledge_base.is_rephrase_candidate(match):
        await _start_or_continue_rephrase(
            session=session,
            conversation_id=conversation_id,
            question=question,
            send_reply=send_reply,
            escalate=escalate,
        )
        return

    await escalate(
        EscalationReason.LLM_CANNOT_ANSWER,
        category=match.category,
        question_text=question,
    )


@dataclass(slots=True)
class _ResolvedPart:
    match: MatchResult
    answer: str | None = None
    answer_category: str | None = None
    result_type: RequestResult | None = None


async def _resolve_llm_part(
    *,
    match: MatchResult,
    question: str,
    conversation_id: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
) -> _ResolvedPart:
    """Resolve an LLM-backed subquestion without sending/escalating."""
    candidates = match.relevant_entries
    candidate_ids = {str(e.get("id")) for e in candidates if e.get("id")}

    if not candidate_ids:
        return _ResolvedPart(match=match)

    result = await llm_client.classify(
        question=question,
        relevant_entries=candidates,
    )

    if result.is_noise:
        return _ResolvedPart(match=match)

    if result.can_answer:
        selected_kb_id = str(result.selected_kb_id or "")
        if selected_kb_id not in candidate_ids or not result.answer:
            logger.warning(
                "Invalid LLM multi-part result | conv=%s | selected=%s | candidates=%s",
                conversation_id,
                selected_kb_id,
                sorted(candidate_ids),
            )
            return _ResolvedPart(match=match)

        entry = knowledge_base.get_by_id(selected_kb_id)
        return _ResolvedPart(
            match=match,
            answer=result.answer,
            answer_category=entry.get("category") if entry else match.category,
            result_type=RequestResult.LLM,
        )

    # Stateful clarification inside a multi-question message remains conservative.
    if result.needs_clarification:
        return _ResolvedPart(match=match)

    return _ResolvedPart(match=match)


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
    """Resolve independent parts while preserving successful answers.

    Current policy intentionally keeps unresolved parts as operator escalations.
    It does not yet implement the future user-facing top-k clarification/rephrase flow.
    """
    resolved_answers: list[tuple[str, str | None, RequestResult]] = []
    unresolved: list[MatchResult] = []
    clarification_parts: list[MatchResult] = []
    operator_parts: list[MatchResult] = []
    seen_categories: set[str] = set()

    for match in matches:
        part = match.subquestion or original_question

        if match.mode == MatchMode.TEMPLATE and match.template_answer:
            category_key = match.kb_id or match.category or f"template:{len(resolved_answers)}"
            if category_key not in seen_categories:
                resolved_answers.append(
                    (
                        match.template_answer,
                        match.category,
                        RequestResult.TEMPLATE,
                    )
                )
                seen_categories.add(category_key)
            continue

        if match.mode == MatchMode.CLARIFICATION:
            clarification_parts.append(match)
            continue

        if match.mode == MatchMode.OPERATOR:
            operator_parts.append(match)
            continue

        if match.mode == MatchMode.NO_MATCH:
            unresolved.append(match)
            continue

        if match.mode == MatchMode.LLM:
            resolved = await _resolve_llm_part(
                match=match,
                question=part,
                conversation_id=conversation_id,
                llm_client=llm_client,
                knowledge_base=knowledge_base,
            )
            if resolved.answer and resolved.result_type:
                resolved_answers.append(
                    (
                        resolved.answer,
                        resolved.answer_category,
                        resolved.result_type,
                    )
                )
            else:
                unresolved.append(match)

    if operator_parts:
        # User explicitly requested operator in any part; do not hide that intent.
        await escalate(
            EscalationReason.USER_REQUESTED_OPERATOR,
            category=operator_parts[0].category,
            question_text=operator_parts[0].subquestion or original_question,
        )
        return

    if clarification_parts:
        # Keep the existing one-at-a-time stateful clarification semantics.
        # We can safely send already-resolved answers first.
        if len(clarification_parts) == 1 and not unresolved:
            clarification = clarification_parts[0]
            if not clarification.kb_id or not clarification.template_answer:
                await escalate(
                    EscalationReason.INTERNAL_ERROR,
                    category=clarification.category,
                    question_text=original_question,
                )
                return

            response_parts = [x[0] for x in resolved_answers]
            response_parts.append(clarification.template_answer)
            await send_reply("\n\n".join(response_parts))
            await save_pending_clarification(
                session=session,
                conversation_id=conversation_id,
                category_id=clarification.kb_id,
            )
            await save_result(
                RequestResult.TEMPLATE,
                category="multiple",
                answer="\n\n".join(response_parts),
                question_text=original_question,
            )
            return

        # Multiple independent state transitions cannot be represented safely
        # by the current pending-clarification storage.
        unresolved.extend(clarification_parts)

    if resolved_answers:
        response = "\n\n".join(x[0] for x in resolved_answers)
        await send_reply(response)
        await save_result(
            RequestResult.LLM if any(x[2] == RequestResult.LLM for x in resolved_answers)
            else RequestResult.TEMPLATE,
            category="multiple",
            answer=response,
            question_text=original_question,
        )

    if unresolved:
        unique: list[tuple[str, str | None, MatchResult]] = []
        seen: set[str] = set()
        for item in unresolved:
            text = item.subquestion or original_question
            if text not in seen:
                unique.append((text, item.category, item))
                seen.add(text)

        # Handle one unresolved subquestion at a time. This keeps the
        # conversational state unambiguous.
        for text, category, item in unique:
            if await _maybe_start_category_choice(
                session=session,
                conversation_id=conversation_id,
                question=text,
                knowledge_base=knowledge_base,
                send_reply=send_reply,
                save_result=save_result,
            ):
                if resolved_answers:
                    await send_reply("\n\n".join(x[0] for x in resolved_answers))
                return

            if knowledge_base.is_rephrase_candidate(item):
                if resolved_answers:
                    await send_reply(
                        "\n\n".join(x[0] for x in resolved_answers)
                        + "\n\nНе удалось точно понять ещё один вопрос. "
                        "Пожалуйста, переформулируйте его подробнее."
                    )
                    await save_result(
                        RequestResult.TEMPLATE,
                        category="multiple",
                        answer="\n\n".join(x[0] for x in resolved_answers),
                        question_text=original_question,
                    )
                    await register_failure(
                        session=session,
                        conversation_id=conversation_id,
                        original_question=text,
                    )
                    return

                await _start_or_continue_rephrase(
                    session=session,
                    conversation_id=conversation_id,
                    question=text,
                    send_reply=send_reply,
                    escalate=escalate,
                )
                return

            await escalate(
                EscalationReason.NO_CATEGORY,
                category=category,
                question_text=text,
            )

    if not resolved_answers and not unresolved:
        await escalate(
            EscalationReason.NO_CATEGORY,
            question_text=original_question,
        )
