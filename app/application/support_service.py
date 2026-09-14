from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.domain.enums import Channel, ConversationStateKind, Decision, EscalationReason
from app.core.config import settings
from app.domain.models import InboundMessage
from app.domain.policy import escalation_ack
from app.domain.response_composer import ResponseComposer
from app.application.decision_pipeline import DecisionPipeline
from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.persistence.repository import (
    EventRepository,
    EscalatedConversationRepository,
    StateRepository,
    InteractionRepository,
    ConversationTurnRepository,
)
from app.infrastructure.persistence.models import InboundEvent
from app.infrastructure.persistence.outbox import OutboundRepository
from app.infrastructure.telegram.notifier import TelegramNotifier
from app.infrastructure.ai.knowledge_base import contains_operator_request, is_non_query_message

logger = logging.getLogger("support_service")

_CLARIFICATION_STATES = {ConversationStateKind.CLARIFYING, ConversationStateKind.REPHRASING}


class SupportApplication:
    def __init__(
        self,
        pipeline: DecisionPipeline,
        intercom: IntercomGateway,
        telegram: TelegramNotifier | None = None,
        outbox: OutboundRepository | None = None,
    ):
        self.pipeline = pipeline
        self.intercom = intercom
        self.telegram = telegram
        self.outbox = outbox

    def _outbox(self, session) -> OutboundRepository:
        return self.outbox or OutboundRepository(session)

    async def _enqueue_reply(
        self,
        session,
        message: InboundMessage,
        event: InboundEvent,
        text: str,
        *,
        suffix: str = "reply",
        memory_items: list[dict[str, str]] | None = None,
    ) -> None:
        if not text:
            return
        action_key = f"event:{event.id}:{suffix}"
        turn_id = None
        if hasattr(session, "scalar") and hasattr(session, "flush"):
            turn = await ConversationTurnRepository(session).ensure_assistant_turn(
                entity_id=message.entity_id,
                channel=message.channel.value,
                text=text,
                event_id=event.id,
                action_key=action_key,
                status="pending",
            )
            turn_id = turn.id
        await self._outbox(session).enqueue(
            action_key=action_key,
            action_type="intercom_reply",
            entity_id=message.entity_id,
            channel=message.channel.value,
            event_id=event.id,
            payload={
                "entity_id": message.entity_id,
                "channel": message.channel.value,
                "text": text,
                "_conversation_turn_id": turn_id,
                "_conversation_turn_action_key": action_key,
                "_routing_memory": list(memory_items or []),
            },
        )

    async def _enqueue_telegram_escalation(
        self,
        session,
        message: InboundMessage,
        event: InboundEvent,
        reason: str,
    ) -> None:
        await self._outbox(session).enqueue(
            action_key=f"event:{event.id}:telegram",
            action_type="telegram_escalation",
            entity_id=message.entity_id,
            channel=message.channel.value,
            event_id=event.id,
            payload={
                "entity_id": message.entity_id,
                "channel": message.channel.value,
                "reason": reason,
                "text": message.text,
                "user_name": message.user_name,
                "user_email": message.user_email,
                "updated_at": message.updated_at,
            },
        )

    async def _escalate(
        self,
        session,
        message: InboundMessage,
        event: InboundEvent | None,
        reason: EscalationReason,
        *,
        reply_text: str | None = None,
    ) -> None:
        if event is None:
            raise RuntimeError("Escalation requires an inbound event")

        states = StateRepository(session)
        escalations = EscalatedConversationRepository(session)

        await states.replace(message.entity_id, ConversationStateKind.ESCALATED, {
            "reason": reason.value,
        })
        await escalations.record(
            conversation_id=message.entity_id,
            channel=message.channel.value,
            reason=reason.value,
            user_name=message.user_name,
            user_email=message.user_email,
            last_message_text=message.text,
            last_message_at=message.updated_at,
            source_event_id=event.id,
        )

        # All external writes are transactional-outbox actions. The escalation
        # blocklist is committed together with them, so a later retry can never
        # re-enter the bot pipeline merely because Intercom/Telegram was unavailable.
        await self._enqueue_reply(
            session,
            message,
            event,
            reply_text or ResponseComposer.escalation_ack(escalation_ack()),
            suffix="escalation_ack",
        )
        await self._outbox(session).enqueue(
            action_key=f"event:{event.id}:escalation_tag",
            action_type="intercom_escalation_tag",
            entity_id=message.entity_id,
            channel=message.channel.value,
            event_id=event.id,
            payload={
                "entity_id": message.entity_id,
                "channel": message.channel.value,
            },
        )
        await self._enqueue_telegram_escalation(session, message, event, reason.value)

    @staticmethod
    def _attempts_after_failed_clarification(state) -> int:
        """Return consecutive clarification requests already consumed."""
        data = state.data if isinstance(getattr(state, "data", None), dict) else {}
        raw = data.get("clarification_attempts", data.get("attempts", 0))
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _short_reply_token(text: str) -> str:
        import re
        value = (text or "").strip().lower().replace("ё", "е")
        value = re.sub(r"[^a-zа-я0-9\- ]+", " ", value)
        return " ".join(value.split())

    async def _resolve_short_clarification_reply(
        self, session, event: InboundEvent, message: InboundMessage, state
    ) -> bool:
        if state.state != ConversationStateKind.CLARIFYING:
            return False
        token = self._short_reply_token(message.text)
        if len(token.split()) != 1:
            return False

        data = state.data if isinstance(state.data, dict) else {}
        kb_id = data.get("kb_id")
        routes = self.pipeline.kb.get_short_clarification_routes(kb_id)
        target_id = routes.get(token)
        if not target_id:
            return False

        entry = self.pipeline.kb.get_by_id(target_id)
        answer = str((entry or {}).get("template_answer") or "").strip()
        if not answer:
            # A contextual reply is only accepted when the target itself is
            # routable; otherwise silently leave it for the generic gate.
            return False

        await self._enqueue_reply(
            session, message, event, ResponseComposer.final_answer([answer]),
            suffix="clarification_choice_answer",
        )
        await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
        return True

    async def _handle_state(self, session, event: InboundEvent, message: InboundMessage, state) -> bool:
        try:
            normalized_state = ConversationStateKind(state.state)
        except (TypeError, ValueError):
            # Corrupt/unknown state must not silently consume the event. Reset to
            # the safe root state and let the normal classifier recover the turn.
            logger.error(
                "Invalid conversation state entity=%s state=%r; resetting to IDLE",
                message.entity_id, state.state,
            )
            await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
            return False

        if normalized_state in _CLARIFICATION_STATES:
            if await self._resolve_short_clarification_reply(session, event, message, state):
                return True
            # Acknowledgements must never turn an exhausted clarification flow into
            # a pointless operator hand-off. They are not substantive turns; keep the
            # pending state intact and wait for the user's next real answer.
            if is_non_query_message(message.text):
                return True
            # The initial clarification/rephrase already counts as request #1.
            # Two failed follow-up messages may receive request #2; the next failed
            # message is escalated before the pipeline can produce another reply.
            attempts = self._attempts_after_failed_clarification(state)
            if attempts >= max(1, int(settings.MAX_CLARIFICATION_ATTEMPTS)):
                await self._escalate(
                    session,
                    message,
                    event,
                    EscalationReason.CLARIFICATION_EXHAUSTED,
                )
                return True

            # Do not consume the message here. Let the existing routing pipeline try
            # to understand the user's clarification. A successful answer resets the
            # state to IDLE; another clarification increments the counter below.
            return False

        if normalized_state == ConversationStateKind.CHOOSING:
            # Persisted menus are part of the state-machine contract. Validate the
            # shape before using it; a malformed menu must never reach the numeric
            # parser or produce a misleading answer.
            data = state.data if isinstance(state.data, dict) else {}
            candidate_ids = data.get("candidate_ids")
            snapshots = data.get("candidates")
            if (
                not isinstance(candidate_ids, list)
                or not 2 <= len(candidate_ids) <= 3
                or any(not str(item).strip() for item in candidate_ids)
                or len({str(item).strip() for item in candidate_ids}) != len(candidate_ids)
                or (snapshots is not None and not isinstance(snapshots, list))
            ):
                logger.error(
                    "Invalid CHOOSING state entity=%s data=%r; resetting to IDLE",
                    message.entity_id, state.data,
                )
                await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
                return False

            # Defensive duplicate of the global operator gate. Keeping this
            # guard makes `_handle_state` safe when exercised directly and when
            # called by any future internal workflow that bypasses `handle()`.
            if contains_operator_request(message.text):
                await self._escalate(
                    session,
                    message,
                    event,
                    EscalationReason.USER_REQUESTED_OPERATOR,
                )
                return True

            candidate_ids = state.data.get("candidate_ids", [])
            # Acknowledgements such as "спасибо" are not a new support request.
            # CHOOSING is necessarily a later conversation turn, so the same
            # curated ignore policy applies here without consuming menu state.
            if is_non_query_message(message.text):
                return True

            # Deterministic numeric/ordinal selection is resolved BEFORE any
            # generic classifier call. The persisted candidate snapshot is the
            # source of truth for the answer, not the current KB ranking.
            indices = self.pipeline.kb.parse_category_choice(
                message.text,
                max_index=len(candidate_ids),
            )
            if not indices:
                # Free text is a real support turn, not an invalid menu action.
                # Do not call the pipeline here and then call it again in handle():
                # that would perform duplicate LLM/retrieval work and could produce
                # two different decisions for the same message. Returning False
                # lets the normal pipeline process this turn exactly once.
                # The pending choice is preserved unless the classifier produces a
                # new ANSWER/CHOICE/CLARIFY outcome; an unresolved message is
                # escalated by the pipeline rather than silently consumed.
                return False

            snapshots = list(state.data.get("candidates") or [])
            answers = []
            for idx in indices:
                if 1 <= idx <= len(candidate_ids):
                    answer = ""
                    if idx <= len(snapshots) and isinstance(snapshots[idx - 1], dict):
                        answer = str(snapshots[idx - 1].get("answer") or "")
                    if not answer:
                        entry = self.pipeline.kb.get_by_id(candidate_ids[idx - 1])
                        answer = str((entry or {}).get("template_answer") or "")
                    if answer:
                        answers.append(answer)
            if not answers:
                await self._escalate(
                    session,
                    message,
                    event,
                    EscalationReason.NO_CATEGORY,
                )
                return True

            await self._enqueue_reply(
                session,
                message,
                event,
                ResponseComposer.final_answer(answers),
                suffix="choice_answer",
            )
            await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
            return True

        if normalized_state == ConversationStateKind.ESCALATED:
            # Operator owns the thread after escalation; do not create a competing bot reply.
            return True

        return False

    @staticmethod
    def _routing_context_for_state(state) -> list[dict[str, str]]:
        """Expose pending state and compact answered-intent memory to the router.

        State metadata is explicitly separated from customer-authored USER/ASSISTANT
        turns. The memory helps the router avoid repeating already answered content,
        while still allowing a new question within the same intent.
        """
        try:
            kind = ConversationStateKind(state.state)
        except (TypeError, ValueError):
            return []
        data = state.data if isinstance(state.data, dict) else {}
        result: list[dict[str, str]] = []

        memory = data.get("_routing_memory")
        if isinstance(memory, list):
            compact: list[dict[str, str]] = []
            for item in memory[-8:]:
                if not isinstance(item, dict):
                    continue
                kb_id = str(item.get("kb_id") or "").strip()
                question = str(item.get("question") or "").strip()
                answer = str(item.get("answer") or "").strip()
                if kb_id and answer:
                    compact.append({
                        "kb_id": kb_id,
                        "question": question[-240:],
                        "answer": answer[-500:],
                    })
            if compact:
                import json
                result.append({
                    "role": "system",
                    "text": (
                        "ROUTING_MEMORY: previously sent Terra answers. "
                        "Use this only to avoid repeating already closed content; a new "
                        "aspect of the same intent may still be answered. "
                        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
                    ),
                })

        if kind == ConversationStateKind.CLARIFYING:
            result.append({
                "role": "system",
                "text": (
                    "PENDING_CLARIFICATION: kb_id={kb}; attempts={attempts}; original_question={question}"
                    .format(
                        kb=str(data.get("kb_id") or ""),
                        attempts=str(data.get("clarification_attempts", data.get("attempts", 0)) or 0),
                        question=str(data.get("original_question") or ""),
                    )
                ),
            })
        elif kind == ConversationStateKind.CHOOSING:
            ids = [str(x).strip() for x in (data.get("candidate_ids") or []) if str(x).strip()]
            original = str(data.get("original_question") or "")
            result.append({
                "role": "system",
                "text": (
                    "PENDING_CHOICE: candidate_ids={ids}; original_question={question}"
                    .format(ids=", ".join(ids), question=original)
                ),
            })
        elif kind == ConversationStateKind.REPHRASING:
            result.append({
                "role": "system",
                "text": (
                    "PENDING_REPHRASE: attempts={attempts}; original_question={question}. "
                    "The previous Terra route did not safely close the request. "
                    "Treat the current user turn as a rephrased attempt to resolve that "
                    "same unresolved request, using the original question and prior "
                    "assistant answer context before choosing a new intent."
                ).format(
                    attempts=str(data.get("attempts", data.get("clarification_attempts", 0)) or 0),
                    question=str(data.get("original_question") or ""),
                ),
            })
        return result

    @staticmethod
    def _remember_business_answers(state, traces, business_answers: list[str]) -> None:
        """Persist compact intent→answer memory for later routing decisions.

        One composed customer-facing answer may close several intents in a compound
        turn. The exact sent answer is therefore associated with every selected intent.
        """
        if not business_answers:
            return
        data = dict(state.data) if isinstance(state.data, dict) else {}
        memory = list(data.get("_routing_memory") or []) if isinstance(data.get("_routing_memory"), list) else []

        intent_questions: dict[str, str] = {}
        ordered_ids: list[str] = []
        for trace in traces:
            ids = [str(x).strip() for x in (getattr(trace, "selected_kb_ids", None) or []) if str(x).strip()]
            if not ids and getattr(trace, "kb_id", None):
                ids = [str(trace.kb_id).strip()]
            question = str(getattr(trace, "question", "") or "")[-500:]
            for kb_id in ids:
                if not kb_id:
                    continue
                if kb_id not in intent_questions:
                    ordered_ids.append(kb_id)
                    intent_questions[kb_id] = question

        if not ordered_ids:
            return

        answers = [str(answer or "").strip() for answer in business_answers if str(answer or "").strip()]
        if not answers:
            return
        # The application normally passes the single composed message. Keep a safe
        # fallback for older callers that still pass multiple fragments.
        answer = answers[0] if len(answers) == 1 else "\n\n".join(answers)
        for kb_id in ordered_ids:
            memory.append({
                "kb_id": kb_id,
                "question": intent_questions[kb_id],
                "answer": answer[-1500:],
            })

        data["_routing_memory"] = memory[-20:]
        state.data = data


    @staticmethod
    def _should_reset_state_after_result(current_state, traces) -> bool:
        """Whether a completed turn may clear persisted conversational state.

        IGNORE is intentionally non-consuming while a clarification/rephrase is
        pending: acknowledgements must not erase the routing anchor needed for the
        next substantive answer. In IDLE/CHOOSING, IGNORE may safely leave state as
        -is because those paths are already handled by their own state machine rules.
        """
        stateful = {Decision.CLARIFY, Decision.CHOICE, Decision.REPHRASE}
        if any(t.decision in stateful for t in traces):
            return False
        if all(t.decision == Decision.IGNORE for t in traces):
            try:
                pending = ConversationStateKind(current_state) in _CLARIFICATION_STATES
            except (TypeError, ValueError):
                pending = False
            return not pending
        return True

    async def handle(
        self,
        session,
        event: InboundEvent,
        message: InboundMessage,
    ) -> bool:
        escalations = EscalatedConversationRepository(session)
        if await escalations.is_escalated(message.entity_id):
            logger.info("Ignoring event for escalated conversation entity=%s", message.entity_id)
            return True

        # Strict routing order is an invariant:
        # 1) active escalation blocklist; 2) state row lock + event ordering;
        # 3) explicit operator request; 4) persisted conversational state;
        # 5) fresh classifier.
        # Ordering must happen BEFORE the operator fast-path so a delayed old event
        # cannot escalate a conversation that has already advanced on a newer event.
        states = StateRepository(session)
        # Serialize all stateful decisions for a conversation. Without a row lock,
        # two workers can process two inbound messages for the same conversation
        # concurrently and both observe the same clarification counter.
        state = await states.get(message.entity_id, for_update=True)

        # Intercom does not guarantee webhook delivery order. The conversation
        # state row is already locked above, so store the provider action time in
        # the same transaction before any stateful branch can return. A later
        # event therefore establishes a monotonic per-conversation watermark.
        # Equal timestamps are deliberately accepted because the provider does not
        # provide a reliable tie-breaker; row locking serializes those events.
        incoming_order_at = getattr(event, "event_order_at", None)
        state_data = state.data if isinstance(state.data, dict) else {}
        ordering = state_data.get("_ordering") if isinstance(state_data.get("_ordering"), dict) else {}
        last_order_at = ordering.get("last_event_order_at")
        if incoming_order_at is not None and last_order_at is not None:
            try:
                if int(incoming_order_at) < int(last_order_at):
                    logger.info(
                        "stale_webhook_ignored entity=%s event=%s order_at=%s last_order_at=%s",
                        message.entity_id, event.id, incoming_order_at, last_order_at,
                    )
                    return False
            except (TypeError, ValueError):
                logger.warning("invalid event ordering metadata entity=%s event=%s", message.entity_id, event.id)

        if incoming_order_at is not None and (last_order_at is None or int(incoming_order_at) >= int(last_order_at)):
            state_data = dict(state_data)
            state_data["_ordering"] = {
                **ordering,
                "last_event_order_at": int(incoming_order_at),
            }
            state.data = state_data
            await session.flush()

        provider_part_id = None
        if isinstance(message.raw, dict):
            provider_part_id = message.raw.get("_provider_part_id")
        try:
            turn_at = datetime.fromtimestamp(int(message.updated_at), tz=timezone.utc) if str(message.updated_at or "").isdigit() else None
        except (TypeError, ValueError, OverflowError):
            turn_at = None
        transcript_repo = None
        # The production AsyncSession supports scalar/flush. Lightweight unit-test
        # doubles from older releases do not; transcript persistence is additive and
        # must never become a prerequisite for routing.
        if hasattr(session, "scalar") and hasattr(session, "flush"):
            transcript_repo = ConversationTurnRepository(session)
            await transcript_repo.ensure_user_turn(
                entity_id=message.entity_id,
                channel=message.channel.value,
                text=message.text,
                event_id=event.id,
                provider_part_id=str(provider_part_id) if provider_part_id else None,
                turn_at=turn_at,
            )

        if contains_operator_request(message.text):
            await self._escalate(
                session,
                message,
                event,
                EscalationReason.USER_REQUESTED_OPERATOR,
            )
            return True

        if await self._handle_state(session, event, message, state):
            return True

        # A state-row transition and the permanent escalation table are both checked
        # defensively. This makes ESCALATED safe even if an older deployment left only
        # conversation_states populated.
        if state.state == ConversationStateKind.ESCALATED:
            await escalations.record(
                conversation_id=message.entity_id,
                channel=message.channel.value,
                reason=str(state.data.get("reason") or EscalationReason.NO_CATEGORY.value),
                user_name=message.user_name,
                user_email=message.user_email,
                last_message_text=message.text,
                last_message_at=message.updated_at,
                source_event_id=event.id,
            )
            return True

        # The current inbound event is already stored before the worker reaches
        # this point. Therefore count==1 means this is the first user turn.
        # Ignore is allowed only on later turns; short domain words must still
        # reach the AI-first pipeline.
        try:
            # Count already-persisted user turns, excluding this event. Looking at
            # inbound_events is incorrect when several customer webhooks are queued
            # before the first one is processed: the first turn could then be treated
            # as a later turn merely because a newer event already exists in the DB.
            previous_user_turns = await transcript_repo.count_user_webhook_turns(
                message.entity_id, exclude_event_id=event.id
            ) if transcript_repo is not None else 0
            is_first_message = previous_user_turns == 0
        except AttributeError:
            # Lightweight test doubles / non-SQL sessions may not expose transcript
            # counting. Fall back to the durable inbound-event count in tests/legacy.
            try:
                inbound_count = await EventRepository(session).count_customer_events_for_entity(message.entity_id)
                is_first_message = inbound_count <= 1
            except AttributeError:
                data = state.data if isinstance(getattr(state, "data", None), dict) else {}
                is_first_message = not bool(data.get("seen_user_message"))
        context = message.raw.get("_conversation_context", []) if isinstance(message.raw, dict) else []
        if not isinstance(context, list):
            context = []

        # Intercom webhooks frequently contain only the triggering conversation
        # part, not the complete dialogue history. The LLM router must nevertheless
        # understand short follow-ups such as "код"/"не приходит" in context.
        # Fetch the authoritative conversation/ticket resource when available and
        # fall back to webhook-local context if the read is temporarily unavailable.
        provider_part_id = None
        if isinstance(message.raw, dict):
            provider_part_id = message.raw.get("_provider_part_id")
        try:
            before_epoch = int(message.updated_at) if str(message.updated_at or "").isdigit() else None
        except (TypeError, ValueError):
            before_epoch = None

        context_fetch_started = time.perf_counter()
        provider_context_fetch_succeeded = False
        provider_context: list[dict[str, str]] = []
        try:
            if settings.FETCH_FULL_CONTEXT and message.channel == Channel.CONVERSATION and hasattr(self.intercom, "get_conversation_context"):
                fetched = await self.intercom.get_conversation_context(
                    message.entity_id,
                    current_part_id=str(provider_part_id) if provider_part_id else None,
                    before_epoch=before_epoch,
                    limit=int(getattr(settings, "LLM_CONTEXT_TURNS", 50)),
                )
                provider_context_fetch_succeeded = True
                provider_context = list(fetched or [])
                if fetched:
                    context = fetched
                logger.info(
                    "conversation_context_fetched | entity=%s | event=%s | messages=%d | elapsed_ms=%.1f",
                    message.entity_id, event.id, len(fetched),
                    (time.perf_counter() - context_fetch_started) * 1000,
                )
            elif settings.FETCH_FULL_CONTEXT and message.channel == Channel.TICKET and hasattr(self.intercom, "get_ticket_context"):
                fetched = await self.intercom.get_ticket_context(
                    message.entity_id,
                    current_part_id=str(provider_part_id) if provider_part_id else None,
                    before_epoch=before_epoch,
                    limit=int(getattr(settings, "LLM_CONTEXT_TURNS", 50)),
                )
                provider_context_fetch_succeeded = True
                provider_context = list(fetched or [])
                if fetched:
                    context = fetched
                logger.info(
                    "ticket_context_fetched | entity=%s | event=%s | messages=%d | elapsed_ms=%.1f",
                    message.entity_id, event.id, len(fetched),
                    (time.perf_counter() - context_fetch_started) * 1000,
                )
        except Exception as exc:
            # Context enrichment is best-effort. A read failure must not turn an
            # otherwise valid current message into a silent drop; the bounded
            # webhook context remains available to the classifier.
            logger.warning(
                "context_fetch_failed | entity=%s | event=%s | channel=%s | error_type=%s | error=%s",
                message.entity_id, event.id, message.channel.value, type(exc).__name__, exc,
            )

        # Persist successful provider history into the local transcript on every
        # successful fetch, not just on first use. A transient Intercom read failure
        # can therefore be healed by the next successful fetch instead of permanently
        # leaving a gap in the local context. Stable import keys make this idempotent.
        durable_context: list[dict[str, str]] = []
        if transcript_repo is not None:
            if provider_context_fetch_succeeded and provider_context:
                await transcript_repo.import_context(
                    entity_id=message.entity_id,
                    channel=message.channel.value,
                    context=provider_context,
                    current_turn_at=turn_at,
                )
            durable_context = await transcript_repo.recent_context(
                message.entity_id, limit=int(getattr(settings, "LLM_CONTEXT_TURNS", 50))
            )
            # Exclude the current user turn; it is supplied separately as
            # `current_question`, so the prompt has an unambiguous boundary.
            if durable_context:
                last = durable_context[-1]
                if last.get("role") == "user" and last.get("text") == message.text:
                    durable_context = durable_context[:-1]

        # If transcript persistence is unavailable (test adapter or a legacy DB),
        # retain the provider/webhook context rather than making it mandatory.
        # A successful Intercom read can legitimately be empty during replication
        # or when the resource omits older parts. Do not let that empty result
        # erase the bounded history that arrived in the webhook: short follow-ups
        # such as "не пришёл" need that immediately available context. Durable
        # local history remains preferred whenever it actually contains turns.
        context = (
            durable_context or list(context)
            if transcript_repo is not None
            else list(context)
        )
        context = context[-int(getattr(settings, "LLM_CONTEXT_TURNS", 50)):] + self._routing_context_for_state(state)
        try:
            result = await self.pipeline.decide(
                message,
                is_first_message=is_first_message,
                conversation_context=context,
            )
        except TypeError as exc:
            # Compatibility for small legacy pipeline adapters. The production
            # pipeline supports both context arguments.
            if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                raise
            result = await self.pipeline.decide(message)
        logger.info(
            "pipeline_result | entity=%s | event=%s | latency_ms=%.1f | "
            "actions=%d | traces=%s",
            message.entity_id, event.id, float(result.latency_ms or 0.0), len(result.actions),
            [
                {
                    "decision": getattr(t.decision, "value", str(t.decision)),
                    "kb_id": t.kb_id,
                    "category": t.category,
                    "reason": t.decision_reason,
                    "llm_used": t.llm_used,
                    "escalation_reason": getattr(t.escalation_reason, "value", t.escalation_reason),
                }
                for t in result.traces
            ],
        )

        # Application-level delivery invariant: the only legitimate empty-action
        # result is a contextual IGNORE. Any other empty result is a routing defect
        # or an integration failure and must become a visible human hand-off. This
        # guard lives above the classifier so future modules cannot accidentally
        # introduce a new silent path.
        non_ignore_traces = [t for t in result.traces if t.decision != Decision.IGNORE]
        if not result.actions and non_ignore_traces:
            logger.error(
                "empty_result_before_outbox | entity=%s | event=%s | decisions=%s",
                message.entity_id, event.id,
                [getattr(t.decision, "value", str(t.decision)) for t in result.traces],
            )
            result.traces.append(DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="support_empty_result_guard",
                escalation_reason=EscalationReason.INTERNAL_ERROR,
                automated=True,
            ))
            result.actions = [OutboundAction("reply", ResponseComposer.escalation_ack(escalation_ack()), True)]

        interaction_repo = InteractionRepository(session)
        await interaction_repo.add_many(event.id, message.entity_id, message.channel.value, result)

        current_state = state.state
        current_attempts = self._attempts_after_failed_clarification(state)

        # Persist conversational state produced by the policy.
        #
        # A single inbound message can produce several traces (match_many splits
        # compound questions into independent sub-questions). Only ONE pending
        # state can be stored per conversation, but all stateful traces still get
        # their reply text concatenated into the same outgoing message — so if we
        # naively looped and called states.replace() once per trace, whichever
        # trace happened to be last would silently overwrite an earlier one.
        # Concretely: sub-question #1 resolves to CATEGORY_CHOICE (user is shown
        # a numbered Top-3 menu and CHOOSING/candidate_ids is what must be saved),
        # sub-question #2 resolves to REPHRASE — last-write-wins replaced CHOOSING
        # with REPHRASING, the candidate_ids were lost, and the user's next
        # message ("1") was reprocessed as a brand-new query that matches nothing
        # and escalates with no_category.
        #
        # A CHOICE already made an explicit, numbered promise to the user, so it
        # must win over a same-turn CLARIFY/REPHRASE from another sub-question.
        # CLARIFY (tied to a specific kb_id) outranks a generic REPHRASE.
        _STATE_PRIORITY = {Decision.CHOICE: 3, Decision.CLARIFY: 2, Decision.REPHRASE: 1}
        stateful_traces = [t for t in result.traces if t.decision in _STATE_PRIORITY]
        governing_trace = None
        for trace in stateful_traces:
            if governing_trace is None or _STATE_PRIORITY[trace.decision] > _STATE_PRIORITY[governing_trace.decision]:
                governing_trace = trace

        choice_state_persisted = False
        if governing_trace is not None:
            trace = governing_trace
            if trace.decision == Decision.CHOICE:
                # Hard invariant: a user-visible numbered menu is valid only when
                # the exact candidate IDs + answer snapshots are durably persisted
                # before the reply enters the outbox. Never send an unresolvable menu.
                ids = [str(x).strip() for x in (getattr(trace, "choice_candidate_ids", None) or []) if str(x).strip()]
                raw_candidates = list(getattr(trace, "choice_candidates", None) or [])

                if len(raw_candidates) < 2 and len(ids) >= 2:
                    # Defensive compatibility for traces created by an older build.
                    # Reconstruct only from the known IDs and validate every candidate.
                    raw_candidates = []
                    for cid in ids[:3]:
                        entry = self.pipeline.kb.get_by_id(cid) or {}
                        label = str(entry.get("choice_label") or entry.get("category") or cid).strip()
                        answer = str(entry.get("template_answer") or "").strip()
                        if label and answer:
                            raw_candidates.append({"kb_id": cid, "label": label, "answer": answer})

                normalized = []
                seen_ids: set[str] = set()
                for item in raw_candidates[:3]:
                    if not isinstance(item, dict):
                        continue
                    cid = str(item.get("kb_id") or "").strip()
                    label = str(item.get("label") or item.get("choice_label") or cid).strip()
                    answer = str(item.get("answer") or item.get("template_answer") or "").strip()
                    if not cid or not label or not answer or cid in seen_ids:
                        continue
                    normalized.append({"kb_id": cid, "label": label, "answer": answer})
                    seen_ids.add(cid)

                if len(normalized) < 2:
                    logger.error(
                        "choice_state_invalid | entity=%s | ids=%s | candidates=%r",
                        message.entity_id, ids, raw_candidates,
                    )
                    # Do not send the invalid menu. This is a substantive unresolved
                    # query, so use the normal human fallback instead.
                    await self._escalate(
                        session, message, event, EscalationReason.NO_CATEGORY,
                    )
                    return True

                ids = [item["kb_id"] for item in normalized]
                await states.replace(message.entity_id, ConversationStateKind.CHOOSING, {
                    "candidate_ids": ids,
                    "candidates": normalized,
                    "attempts": 0,
                    "original_question": trace.question,
                })
                choice_state_persisted = True
                logger.info(
                    "choice_state_saved | entity=%s | candidate_ids=%s | snapshot_size=%d",
                    message.entity_id, ids, len(normalized),
                )
            else:
                attempts = current_attempts + 1 if current_state in _CLARIFICATION_STATES else 1
                if trace.decision == Decision.CLARIFY and trace.kb_id:
                    await states.replace(message.entity_id, ConversationStateKind.CLARIFYING, {
                        "kb_id": trace.kb_id,
                        "clarification_attempts": attempts,
                        "attempts": attempts,
                        "original_question": trace.question,
                    })
                else:
                    await states.replace(message.entity_id, ConversationStateKind.REPHRASING, {
                        "clarification_attempts": attempts,
                        "attempts": attempts,
                        "original_question": trace.question,
                    })

        if any(t.decision == Decision.CHOICE for t in result.traces) and not choice_state_persisted:
            # Defensive final gate: never allow a Choice action into the outbox unless
            # CHOOSING state has been persisted in this very transaction. If persistence
            # itself failed, the exception propagates and EventWorker retries the event.
            logger.error(
                "choice_state_missing_before_send | entity=%s | event=%s",
                message.entity_id, event.id,
            )
            raise RuntimeError("Top-3 state was not persisted before sending choice")

        # Escalation is persisted BEFORE Telegram notification but in the same database
        # transaction. That guarantees future events stop at the blocklist once the
        # current event is successfully completed.
        escalation_reason = next(
            (t.escalation_reason for t in result.traces if t.escalation_reason),
            None,
        )
        if any(action.escalate for action in result.actions):
            escalation_reply = next(
                (action.text for action in result.actions if action.escalate and action.text),
                None,
            )
            await self._escalate(
                session,
                message,
                event,
                escalation_reason or EscalationReason.NO_CATEGORY,
                reply_text=escalation_reply,
            )
            return True

        # `business_answer` contains only approved business content. It must pass
        # through the response composer before delivery so the user-facing greeting
        # and operator fallback footer are added exactly once. For genuinely
        # compound LLM routes, multiple approved answers are therefore merged into
        # one Intercom message instead of being sent as separate replies.
        business_answers = list(dict.fromkeys(
            action.text for action in result.actions
            if action.kind == "business_answer" and action.text
        ))
        reply_texts = list(dict.fromkeys(
            action.text for action in result.actions
            if action.kind == "reply" and action.text
        ))

        # Keep the exact customer-facing answer in conversation memory before
        # transient state is reset. For a compound turn this is the single composed
        # message actually sent, so every selected intent is associated with the same
        # complete closure of that turn.
        composed_business_answer = ResponseComposer.final_answer(business_answers) if business_answers else ""
        memory_items: list[dict[str, str]] = []
        if composed_business_answer:
            intent_questions: dict[str, str] = {}
            for trace in result.traces:
                ids = [str(x).strip() for x in (getattr(trace, "selected_kb_ids", None) or []) if str(x).strip()]
                if not ids and getattr(trace, "kb_id", None):
                    ids = [str(trace.kb_id).strip()]
                question_text = str(getattr(trace, "question", "") or "")[-500:]
                for kb_id in ids:
                    if kb_id and kb_id not in intent_questions:
                        intent_questions[kb_id] = question_text
            memory_items = [
                {"kb_id": kb_id, "question": question_text, "answer": composed_business_answer[-1500:]}
                for kb_id, question_text in intent_questions.items()
            ]

        reply_index = 0
        if composed_business_answer:
            reply_index += 1
            await self._enqueue_reply(
                session,
                message,
                event,
                composed_business_answer,
                suffix=f"reply:{reply_index}",
                memory_items=memory_items,
            )

        for text in reply_texts:
            reply_index += 1
            await self._enqueue_reply(
                session,
                message,
                event,
                text,
                suffix=f"reply:{reply_index}",
            )

        # A successful non-clarification support outcome ends the clarification
        # flow and resets the counter. IGNORE is different: an acknowledgement such
        # as "спасибо" must not destroy an already-pending clarification state,
        # otherwise the next substantive user answer loses its routing anchor.
        if result.traces and self._should_reset_state_after_result(current_state, result.traces):
            await states.replace(message.entity_id, ConversationStateKind.IDLE, {})
        return True
