from __future__ import annotations

import logging

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
        await self._enqueue_reply(session, message, event, ResponseComposer.escalation_ack(escalation_ack()), suffix="escalation_ack")
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

    async def _handle_state(self, session, event: InboundEvent, message: InboundMessage, state) -> bool:
        if state.state in _CLARIFICATION_STATES:
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

        if state.state == ConversationStateKind.CHOOSING:
            candidate_ids = state.data.get("candidate_ids", [])
            indices = self.pipeline.kb.parse_category_choice(
                message.text,
                max_index=len(candidate_ids),
            )
            if not indices:
                attempts = self._attempts_after_failed_clarification(state)
                if attempts >= 2:
                    await self._escalate(
                        session,
                        message,
                        event,
                        EscalationReason.NO_CATEGORY,
                    )
                else:
                    state.data["attempts"] = attempts + 1
                    await session.flush()
                    await self._enqueue_reply(
                        session,
                        message,
                        event,
                        "Пожалуйста, укажите номер варианта (например, 1 или 2).",
                        suffix="choice_retry",
                    )
                return True

            answers = []
            for idx in indices:
                if 1 <= idx <= len(candidate_ids):
                    entry = self.pipeline.kb.get_by_id(candidate_ids[idx - 1])
                    if entry and entry.get("template_answer"):
                        answers.append(str(entry["template_answer"]))
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

        if state.state == ConversationStateKind.ESCALATED:
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
        for trace in result.traces:
            if trace.decision in {Decision.REPHRASE, Decision.CLARIFY}:
                # The first clarification/rephrase response starts the counter at 1.
                # When this happens while already waiting for clarification, advance
                # the counter rather than resetting it back to 1.
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
            elif trace.decision == Decision.CHOICE:
                try:
                    candidates = self.pipeline.kb.get_category_choice_candidates(trace.question)
                    ids = [str(x["kb_id"]) for x in candidates]
                    if len(ids) >= 2:
                        await states.replace(message.entity_id, ConversationStateKind.CHOOSING, {
                            "candidate_ids": ids,
                            "attempts": 0,
                            "original_question": trace.question,
                        })
                except Exception:
                    logger.exception("Failed to persist category choice state")

        # Escalation is persisted BEFORE Telegram notification but in the same database
        # transaction. That guarantees future events stop at the blocklist once the
        # current event is successfully completed.
        escalation_reason = next(
            (t.escalation_reason for t in result.traces if t.escalation_reason),
            None,
        )
        if any(action.escalate for action in result.actions):
            await self._escalate(
                session,
                message,
                event,
                escalation_reason or EscalationReason.NO_CATEGORY,
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
