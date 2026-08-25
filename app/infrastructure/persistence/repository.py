from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.persistence.models import (
    InboundEvent,
    ConversationState,
    EscalatedConversation,
    SupportInteraction,
)
from app.domain.enums import EventStatus, ConversationStateKind


class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue(self, *, event_key: str, provider_event_id: str, topic: str, entity_id: str, payload: dict) -> bool:
        stmt = (
            pg_insert(InboundEvent)
            .values(
                event_key=event_key,
                provider_event_id=provider_event_id,
                topic=topic,
                entity_id=entity_id,
                payload=payload,
            )
            .on_conflict_do_nothing(index_elements=[InboundEvent.event_key])
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount == 1

    async def claim_one(self) -> InboundEvent | None:
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(minutes=5)
        q = (
            select(InboundEvent)
            .where(
                (
                    (InboundEvent.status == EventStatus.NEW) &
                    (InboundEvent.available_at <= now)
                ) |
                (
                    (InboundEvent.status == EventStatus.PROCESSING) &
                    (InboundEvent.locked_at <= stale_before)
                )
            )
            .order_by(InboundEvent.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        result = await self.session.execute(q)
        event = result.scalar_one_or_none()
        if event:
            event.status = EventStatus.PROCESSING
            event.attempts += 1
            event.locked_at = now
            await self.session.commit()
        return event

    async def mark_done(self, event: InboundEvent) -> None:
        event.status = EventStatus.DONE
        event.processed_at = datetime.now(timezone.utc)
        event.locked_at = None
        event.last_error = None
        await self.session.commit()

    async def mark_failed(
        self,
        event: InboundEvent,
        error: str,
        retry: bool = True,
        *,
        attempt_count: int | None = None,
    ) -> None:
        # Called after a rollback in the worker. Update the row directly so this
        # method never needs to refresh expired ORM state from the rolled-back session.
        if attempt_count is None:
            attempt_count = int(getattr(event, "attempts", 0) or 0)
        else:
            attempt_count = int(attempt_count)
        status = EventStatus.NEW if retry and attempt_count < 5 else EventStatus.FAILED
        values = {
            "last_error": error[:4000],
            "locked_at": None,
            "status": status,
        }
        if status == EventStatus.NEW:
            values["available_at"] = datetime.now(timezone.utc) + timedelta(
                seconds=min(60, 2 ** max(0, attempt_count - 1))
            )
        await self.session.execute(
            update(InboundEvent)
            .where(InboundEvent.id == event.id)
            .values(**values)
        )
        await self.session.commit()


class StateRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, conversation_id: str, *, for_update: bool = False) -> ConversationState:
        if for_update:
            state = await self.session.scalar(
                select(ConversationState)
                .where(ConversationState.conversation_id == conversation_id)
                .with_for_update()
            )
            if state:
                return state

            # The first event for a conversation has no state row yet. Insert it
            # atomically so two workers cannot race into a duplicate-PK error.
            await self.session.execute(
                pg_insert(ConversationState)
                .values(
                    conversation_id=conversation_id,
                    state=ConversationStateKind.IDLE,
                    data={},
                )
                .on_conflict_do_nothing(index_elements=[ConversationState.conversation_id])
            )
            state = await self.session.scalar(
                select(ConversationState)
                .where(ConversationState.conversation_id == conversation_id)
                .with_for_update()
            )
            if state:
                return state
            raise RuntimeError(f"Failed to initialize conversation state: {conversation_id}")

        state = await self.session.get(ConversationState, conversation_id)
        if state:
            return state
        state = ConversationState(conversation_id=conversation_id, state=ConversationStateKind.IDLE, data={})
        self.session.add(state)
        await self.session.flush()
        return state

    async def replace(self, conversation_id: str, state: ConversationStateKind, data: dict) -> None:
        current = await self.get(conversation_id)
        current.state = state
        current.data = data
        await self.session.flush()


class EscalatedConversationRepository:
    """Read/write access to the persistent post-escalation blocklist."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, conversation_id: str) -> EscalatedConversation | None:
        return await self.session.get(EscalatedConversation, conversation_id)

    async def is_escalated(self, conversation_id: str) -> bool:
        return await self.get(conversation_id) is not None

    async def record(
        self,
        *,
        conversation_id: str,
        channel: str,
        reason: str,
        user_name: str = "",
        user_email: str = "",
        last_message_text: str = "",
        last_message_at: str = "",
        source_event_id: int | None = None,
    ) -> EscalatedConversation:
        existing = await self.get(conversation_id)
        if existing:
            # Preserve the original escalation time/reason, but refresh the
            # diagnostic customer information in case the first payload was sparse.
            existing.user_name = user_name or existing.user_name
            existing.user_email = user_email or existing.user_email
            existing.last_message_text = last_message_text or existing.last_message_text
            existing.last_message_at = last_message_at or existing.last_message_at
            existing.source_event_id = source_event_id or existing.source_event_id
            await self.session.flush()
            return existing

        row = EscalatedConversation(
            conversation_id=conversation_id,
            channel=channel,
            reason=reason,
            user_name=user_name,
            user_email=user_email,
            last_message_text=last_message_text,
            last_message_at=last_message_at,
            source_event_id=source_event_id,
        )
        self.session.add(row)
        await self.session.flush()
        return row


class InteractionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_many(self, event_id: int | None, entity_id: str, channel: str, result) -> None:
        for trace in result.traces:
            self.session.add(SupportInteraction(
                event_id=event_id,
                entity_id=entity_id,
                channel=channel,
                question=trace.question,
                decision=trace.decision.value,
                kb_id=trace.kb_id,
                category=trace.category,
                score=trace.score,
                margin=trace.margin,
                query_quality_score=trace.query_quality_score,
                semantic_score=trace.semantic_score,
                lexical_score=trace.lexical_score,
                fuzzy_score=trace.fuzzy_score,
                evidence_score=trace.evidence_score,
                real_intent_evidence=trace.real_intent_evidence,
                embedding_score=trace.embedding_score,
                intent_fit_score=trace.intent_fit_score,
                family_fit_score=trace.family_fit_score,
                domain_support_score=trace.domain_support_score,
                ontology_alignment_score=trace.ontology_alignment_score,
                negative_contrast_score=trace.negative_contrast_score,
                contrastive_margin=trace.contrastive_margin,
                contrastive_competitor_id=trace.contrastive_competitor_id,
                contrastive_reason=trace.contrastive_reason,
                latency_ms=result.latency_ms,
                automated=trace.automated,
                llm_used=trace.llm_used,
                escalation_reason=trace.escalation_reason.value if trace.escalation_reason else None,
                decision_reason=trace.decision_reason,
            ))
        await self.session.flush()
