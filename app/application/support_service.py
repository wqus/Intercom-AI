from __future__ import annotations

import logging
from datetime import datetime

from app.domain.enums import Channel, ConversationStateKind, Decision, EscalationReason
from app.domain.models import InboundMessage
from app.domain.policy import escalation_ack
from app.domain.response_composer import ResponseComposer
from app.application.decision_pipeline import DecisionPipeline
from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.persistence.repository import (
    EscalatedConversationRepository,
    StateRepository,
    InteractionRepository,
)
from app.infrastructure.persistence.models import InboundEvent
from app.infrastructure.persistence.outbox import OutboundRepository
from app.infrastructure.telegram.notifier import TelegramNotifier

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
        """Return the number of failed user attempts accumulated so far."""
        try:
            return max(0, int(state.data.get("attempts", 0)))
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

        kb_id = state.data.get("kb_id")
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

    @staticmethod
    def _message_timestamp(value: str | None) -> float:
        if value in (None, ""):
            return -1.0
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError, OverflowError):
                return -1.0

    async def _handle_state(self, session, event: InboundEvent, message: InboundMessage, state) -> bool:
        try:
            normalized_state = ConversationStateKind(state.state)
        except (TypeError, ValueError):
            # A state not understood by this deployment is not actionable.
            # Never route it as a fresh question and never escalate merely
            # because the state machine has no handler for it.
            logger.warning(
                "Ignoring unsupported conversation state entity=%s state=%r",
                message.entity_id, state.state,
            )
            return True

        if normalized_state in _CLARIFICATION_STATES:
            if await self._resolve_short_clarification_reply(session, event, message, state):
                return True
            # The initial clarification/rephrase already counts as request #1.
            # Two failed follow-up messages may receive request #2; the next failed
            # message is escalated before the pipeline can produce another reply.
            attempts = self._attempts_after_failed_clarification(state)
            if attempts >= 2:
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
            candidate_ids = state.data.get("candidate_ids", [])
            indices = self.pipeline.kb.parse_category_choice(
                message.text,
                max_index=len(candidate_ids),
            )
            if not indices:
                # CHOOSING is a transactional UI contract: until the user selects
                # one of the options, this event must not be allowed to fall through
                # into the generic classifier. Otherwise a delayed/replayed webhook
                # containing the original question can answer that question and reset
                # the menu before the real "1"/"2"/"3" arrives.
                #
                # A genuinely new free-text clarification is still permitted, but only
                # when Intercom provides a timestamp newer than the menu-triggering
                # message. Missing/unparseable timestamps are treated conservatively:
                # keep CHOOSING and ask for the option number.
                prompt_at = self._message_timestamp(state.data.get("prompt_at"))
                incoming_at = self._message_timestamp(message.updated_at)
                is_fresh_after_prompt = (
                    prompt_at >= 0 and incoming_at >= 0 and incoming_at > prompt_at
                )

                if is_fresh_after_prompt:
                    probe = await self.pipeline.decide(message)
                    if any(
                        t.decision in {Decision.ANSWER_TEMPLATE, Decision.CHOICE, Decision.CLARIFY}
                        for t in probe.traces
                    ):
                        return False

                # Never escalate or silently destroy a still-live Top-3 state because
                # of an invalid/non-choice message. Keep the exact candidate snapshot.
                attempts = self._attempts_after_failed_clarification(state)
                state.data["attempts"] = attempts + 1
                await session.flush()
                if attempts == 0:
                    await self._enqueue_reply(
                        session,
                        message,
                        event,
                        "Пожалуйста, укажите номер варианта (например, 1 или 2).",
                        suffix="choice_retry",
                    )
                return True

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

            selected = []
            for idx in indices:
                if 1 <= idx <= len(snapshots) and isinstance(snapshots[idx - 1], dict):
                    item = snapshots[idx - 1]
                    selected.append({
                        "index": idx,
                        "kb_id": str(item.get("kb_id") or ""),
                        "category": str(item.get("category") or ""),
                    })
            logger.info(
                "choice_resolved | entity=%s | selected=%s",
                message.entity_id,
                selected,
            )
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

    async def handle(
        self,
        session,
        event: InboundEvent,
        message: InboundMessage,
    ) -> None:
        escalations = EscalatedConversationRepository(session)
        if await escalations.is_escalated(message.entity_id):
            logger.info("Ignoring event for escalated conversation entity=%s", message.entity_id)
            return

        states = StateRepository(session)
        # Serialize all stateful decisions for a conversation. Without a row lock,
        # two workers can process two inbound messages for the same conversation
        # concurrently and both observe the same clarification counter.
        state = await states.get(message.entity_id, for_update=True)

        if await self._handle_state(session, event, message, state):
            return

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
            return

        result = await self.pipeline.decide(message)
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
                    category = str(item.get("category") or "").strip()
                    answer = str(item.get("answer") or item.get("template_answer") or "").strip()
                    if not cid or not label or not answer or cid in seen_ids:
                        continue
                    normalized.append({
                        "kb_id": cid,
                        "label": label,
                        "category": category,
                        "answer": answer,
                    })
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
                    return

                ids = [item["kb_id"] for item in normalized]
                await states.replace(message.entity_id, ConversationStateKind.CHOOSING, {
                    "candidate_ids": ids,
                    "candidates": normalized,
                    "attempts": 0,
                    "original_question": trace.question,
                    # Timestamp of the user message that caused the menu. This lets
                    # CHOOSING distinguish a genuinely new clarification from a late
                    # replay of an older webhook/history snapshot.
                    "prompt_at": message.updated_at or "",
                    "prompt_event_id": message.event_id or "",
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
                        "attempts": attempts,
                        "original_question": trace.question,
                    })
                else:
                    await states.replace(message.entity_id, ConversationStateKind.REPHRASING, {
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
            return

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

        # A successful non-clarification outcome ends the clarification flow and
        # therefore resets the persisted attempts counter to zero.
        if result.traces and not any(
            t.decision in {Decision.CLARIFY, Decision.CHOICE, Decision.REPHRASE}
            for t in result.traces
        ):
            await states.replace(message.entity_id, ConversationStateKind.IDLE, {})
