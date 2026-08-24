from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.persistence.models import InboundEvent, ConversationState, SupportInteraction
from app.domain.enums import EventStatus, ConversationStateKind

class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue(self, *, event_key: str, provider_event_id: str, topic: str, entity_id: str, payload: dict) -> bool:
        existing = await self.session.scalar(select(InboundEvent).where(InboundEvent.event_key == event_key))
        if existing:
            return False
        self.session.add(InboundEvent(
            event_key=event_key, provider_event_id=provider_event_id,
            topic=topic, entity_id=entity_id, payload=payload,
        ))
        await self.session.commit()
        return True

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

    async def mark_failed(self, event: InboundEvent, error: str, retry: bool = True) -> None:
        event.last_error = error[:4000]
        event.locked_at = None
        event.status = EventStatus.NEW if retry and event.attempts < 5 else EventStatus.FAILED
        if event.status == EventStatus.NEW:
            delay = min(60, 2 ** max(0, event.attempts - 1))
            event.available_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        await self.session.commit()

class StateRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, conversation_id: str) -> ConversationState:
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

