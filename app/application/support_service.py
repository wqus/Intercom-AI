from __future__ import annotations

import logging

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
)
from app.infrastructure.persistence.models import InboundEvent
from app.infrastructure.persistence.outbox import OutboundRepository
from app.infrastructure.telegram.notifier import TelegramNotifier
from app.infrastructure.ai.knowledge_base import contains_operator_request

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
    ) -> None:
        if not text:
            return
        await self._outbox(session).enqueue(
            action_key=f"event:{event.id}:{suffix}",
            action_type="intercom_reply",
            entity_id=message.entity_id,
            channel=message.channel.value,
            event_id=event.id,
            payload={
                "entity_id": message.entity_id,
                "channel": message.channel.value,
                "text": text,
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
            from app.infrastructure.ai.knowledge_base import is_non_query_message
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
            from app.infrastructure.ai.knowledge_base import is_non_query_message
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
        """Expose pending state to the semantic router without changing domain APIs.

        State is routing metadata, not customer-authored conversation text. It is
        therefore represented explicitly and kept compact so the LLM can use the
        pending question/menu without confusing it with a user message.
        """
        try:
            kind = ConversationStateKind(state.state)
        except (TypeError, ValueError):
            return []
        data = state.data if isinstance(state.data, dict) else {}
        if kind == ConversationStateKind.CLARIFYING:
            return [{
                "role": "system",
                "text": (
                    "PENDING_CLARIFICATION: kb_id={kb}; attempts={attempts}; original_question={question}"
                    .format(
                        kb=str(data.get("kb_id") or ""),
                        attempts=str(data.get("clarification_attempts", data.get("attempts", 0)) or 0),
                        question=str(data.get("original_question") or ""),
                    )
                ),
            }]
        if kind == ConversationStateKind.CHOOSING:
            ids = [str(x).strip() for x in (data.get("candidate_ids") or []) if str(x).strip()]
            original = str(data.get("original_question") or "")
            return [{
                "role": "system",
                "text": (
                    "PENDING_CHOICE: candidate_ids={ids}; original_question={question}"
                    .format(ids=", ".join(ids), question=original)
                ),
            }]
        return []


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
        # 1) active escalation blocklist; 2) explicit operator request;
        # 3) persisted conversational state; 4) fresh classifier.
        # A numeric reply such as `1` must never reach the classifier while a
        # CHOOSING state exists, and `оператор` must always escalate directly.
        if contains_operator_request(message.text):
            await self._escalate(
                session,
                message,
                event,
                EscalationReason.USER_REQUESTED_OPERATOR,
            )
            return True

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
            inbound_count = await EventRepository(session).count_for_entity(message.entity_id)
            is_first_message = inbound_count <= 1
        except AttributeError:
            # Lightweight test doubles / non-SQL sessions may not expose scalar().
            # In production the database count above is authoritative.
            data = state.data if isinstance(getattr(state, "data", None), dict) else {}
            is_first_message = not bool(data.get("seen_user_message"))
        context = message.raw.get("_conversation_context", []) if isinstance(message.raw, dict) else []
        if not isinstance(context, list):
            context = []
        # Preserve chronological conversation context and add explicit pending
        # state metadata for clarification/menu continuations. The metadata is
        # never treated as customer-authored text by the prompt.
        context = list(context)[-7:] + self._routing_context_for_state(state)
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

        reply_index = 0
        for action in result.actions:
            if action.kind == "reply" and action.text:
                reply_index += 1
                await self._enqueue_reply(
                    session,
                    message,
                    event,
                    action.text,
                    suffix=f"reply:{reply_index}",
                )

        # A successful non-clarification support outcome ends the clarification
        # flow and resets the counter. IGNORE is different: an acknowledgement such
        # as "спасибо" must not destroy an already-pending clarification state,
        # otherwise the next substantive user answer loses its routing anchor.
        if result.traces and self._should_reset_state_after_result(current_state, result.traces):
            await states.replace(message.entity_id, ConversationStateKind.IDLE, {})
        return True
