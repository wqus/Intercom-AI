import logging
from app.domain.enums import Channel, ConversationStateKind, Decision, EscalationReason
from app.domain.models import InboundMessage
from app.domain.policy import escalation_ack
from app.application.decision_pipeline import DecisionPipeline
from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.persistence.repository import StateRepository, InteractionRepository
from app.infrastructure.persistence.models import InboundEvent
from app.infrastructure.ai.knowledge_base import MatchMode
from app.infrastructure.telegram.notifier import TelegramNotifier

logger = logging.getLogger("support_service")

class SupportApplication:
    def __init__(
        self,
        pipeline: DecisionPipeline,
        intercom: IntercomGateway,
        telegram: TelegramNotifier | None = None,
    ):
        self.pipeline = pipeline
        self.intercom = intercom
        self.telegram = telegram

    async def _deliver(self, message: InboundMessage, text: str, escalate: bool) -> None:
        # Reply delivery is the primary side effect. If it fails, the event must
        # remain retryable. Secondary escalation side effects are best-effort and
        # must never make an already delivered customer reply repeat.
        await self.intercom.send_reply(message.entity_id, message.channel.value, text)
        if not escalate:
            return

        reason = "escalated_to_operator"
        try:
            await self.intercom.add_escalation_tag(message.entity_id, message.channel.value)
        except Exception:
            logger.exception("Escalation tag failed entity=%s", message.entity_id)

        if self.telegram:
            try:
                await self.telegram.notify_escalation(
                    entity_id=message.entity_id,
                    channel=message.channel.value,
                    reason=reason,
                    text=message.text,
                )
            except Exception:
                logger.exception("Operator notification failed entity=%s", message.entity_id)

    async def _handle_state(self, session, message: InboundMessage, state) -> bool:
        if state.state == ConversationStateKind.CLARIFYING:
            kb_id = state.data.get("kb_id")
            normalized = message.text.lower().strip()

            if kb_id == "kb_005":
                wants_new = any(p in normalized for p in ("в новую", "новую школу", "перейти", "сменить"))
                wants_leave = any(p in normalized for p in ("выйти", "покинуть", "просто уйти", "уйти из школы")) and not wants_new

                if wants_new:
                    text=escalation_ack()
                    await self._deliver(message, text, True)
                    await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                        "reason": EscalationReason.WANTS_NEW_SCHOOL.value
                    })
                    return True

                if wants_leave:
                    entry=self.pipeline.kb.get_by_id(kb_id)
                    if entry and entry.get("template_answer"):
                        await self._deliver(message, str(entry["template_answer"]) + (
                            "\n\n---\nЕсли этот ответ не помог решить ваш вопрос, просто напишите «оператор»."
                        ), False)
                        await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
                        return True

            attempts=int(state.data.get("attempts",0))+1
            if attempts >= 2:
                await self._deliver(message, escalation_ack(), True)
                await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                    "reason": EscalationReason.LLM_CANNOT_ANSWER.value
                })
            else:
                state.data["attempts"]=attempts
                await session.flush()
                await self._deliver(
                    message,
                    "Пожалуйста, уточните одним из предложенных вариантов, чтобы я правильно обработал обращение.",
                    False,
                )
            return True

        if state.state == ConversationStateKind.CHOOSING:
            candidate_ids=state.data.get("candidate_ids",[])
            indices=self.pipeline.kb.parse_category_choice(message.text, max_index=len(candidate_ids))
            if not indices:
                attempts=int(state.data.get("attempts",0))+1
                if attempts>=2:
                    await self._deliver(message, escalation_ack(), True)
                    await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                        "reason": EscalationReason.NO_CATEGORY.value
                    })
                else:
                    state.data["attempts"]=attempts
                    await session.flush()
                    await self._deliver(message, "Пожалуйста, укажите номер варианта (например, 1 или 2).", False)
                return True

            answers=[]
            for idx in indices:
                if 1 <= idx <= len(candidate_ids):
                    entry=self.pipeline.kb.get_by_id(candidate_ids[idx-1])
                    if entry and entry.get("template_answer"):
                        answers.append(str(entry["template_answer"]))
            if not answers:
                await self._deliver(message, escalation_ack(), True)
                await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                    "reason": EscalationReason.NO_CATEGORY.value
                })
                return True
            await self._deliver(message, "\n\n".join(answers), False)
            await StateRepository(session).replace(message.entity_id, ConversationStateKind.IDLE, {})
            return True

        if state.state == ConversationStateKind.REPHRASING:
            # Give the user one controlled second attempt; after that, stop consuming
            # the thread and route it to a human.
            attempts=int(state.data.get("attempts", 1))
            if attempts >= 2:
                await self._deliver(message, escalation_ack(), True)
                await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                    "reason": EscalationReason.NO_CATEGORY.value,
                })
                return True
            state.data["attempts"] = attempts + 1
            await session.flush()
            return False

        if state.state == ConversationStateKind.ESCALATED:
            # Operator owns the thread after escalation; don't create a competing bot reply.
            return True

        return False

    async def handle(self, session, event: InboundEvent, message: InboundMessage) -> None:
        states=StateRepository(session)
        state=await states.get(message.entity_id)

        if await self._handle_state(session, message, state):
            return

        result=await self.pipeline.decide(message)
        interaction_repo=InteractionRepository(session)
        await interaction_repo.add_many(event.id, message.entity_id, message.channel.value, result)

        # Persist explicit conversational state produced by the policy.
        for trace in result.traces:
            if trace.decision == Decision.REPHRASE:
                await states.replace(message.entity_id, ConversationStateKind.REPHRASING, {
                    "attempts": 1,
                    "original_question": trace.question,
                })
            elif trace.decision == Decision.CLARIFY and trace.kb_id:
                await states.replace(message.entity_id, ConversationStateKind.CLARIFYING, {
                    "kb_id": trace.kb_id,
                    "attempts": 0,
                    "original_question": trace.question,
                })
            elif trace.decision == Decision.CHOICE:
                try:
                    candidates=self.pipeline.kb.get_category_choice_candidates(trace.question)
                    ids=[str(x["kb_id"]) for x in candidates]
                    if len(ids)>=2:
                        await states.replace(message.entity_id, ConversationStateKind.CHOOSING, {
                            "candidate_ids": ids,
                            "attempts": 0,
                            "original_question": trace.question,
                        })
                except Exception:
                    logger.exception("Failed to persist category choice state")

        for action in result.actions:
            if action.kind == "reply" and action.text:
                await self._deliver(message, action.text, action.escalate)

        if result.traces and not any(t.decision in {Decision.CLARIFY, Decision.CHOICE} for t in result.traces):
            if result.escalated:
                await StateRepository(session).replace(message.entity_id, ConversationStateKind.ESCALATED, {
                    "reason": next((t.escalation_reason.value for t in result.traces if t.escalation_reason), "unknown")
                })
            else:
                await states.replace(message.entity_id, ConversationStateKind.IDLE, {})

