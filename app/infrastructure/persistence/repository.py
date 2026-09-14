from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update, func, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.persistence.models import (
    InboundEvent,
    ConversationState,
    EscalatedConversation,
    SupportInteraction,
    ConversationTurn,
)
from app.domain.enums import EventStatus, ConversationStateKind


class EventRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue(self, *, event_key: str, provider_event_id: str, topic: str, entity_id: str, payload: dict, event_order_at: int | None = None) -> bool:
        stmt = (
            pg_insert(InboundEvent)
            .values(
                event_key=event_key,
                provider_event_id=provider_event_id,
                topic=topic,
                entity_id=entity_id,
                payload=payload,
                event_order_at=event_order_at,
            )
            .on_conflict_do_nothing(index_elements=[InboundEvent.event_key])
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount == 1

    async def count_for_entity(self, entity_id: str) -> int:
        """Return the number of inbound events already recorded for an entity."""
        result = await self.session.scalar(
            select(func.count(InboundEvent.id)).where(InboundEvent.entity_id == entity_id)
        )
        return int(result or 0)

    async def count_customer_events_for_entity(self, entity_id: str) -> int:
        """Count only customer-originated message topics for first-turn semantics.

        Lifecycle events such as ticket.created must not make a later actual customer
        reply look like a non-first turn, especially when a ticket is linked to the
        same Intercom entity as a conversation.
        """
        customer_topics = (
            "conversation.user.created",
            "conversation.user.replied",
            "ticket.contact.replied",
        )
        result = await self.session.scalar(
            select(func.count(InboundEvent.id)).where(
                InboundEvent.entity_id == entity_id,
                InboundEvent.topic.in_(customer_topics),
            )
        )
        return int(result or 0)

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

    async def mark_failed_by_id(
        self,
        event_id: int,
        error: str,
        retry: bool = True,
        *,
        attempt_count: int = 0,
    ) -> None:
        """Mark a failed event without touching an expired ORM instance.

        This method is intentionally keyed by the scalar primary key captured before
        the worker rolls the transaction back. It keeps retry handling independent of
        SQLAlchemy's post-rollback attribute expiration.
        """
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
            .where(InboundEvent.id == int(event_id))
            .values(**values)
        )
        await self.session.commit()

    async def mark_failed(
        self,
        event: InboundEvent,
        error: str,
        retry: bool = True,
        *,
        attempt_count: int | None = None,
    ) -> None:
        """Backward-compatible wrapper for callers that still hold the ORM row."""
        event_id = int(event.id)
        if attempt_count is None:
            attempt_count = int(getattr(event, "attempts", 0) or 0)
        await self.mark_failed_by_id(
            event_id, error, retry=retry, attempt_count=int(attempt_count)
        )


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
        existing_data = current.data if isinstance(current.data, dict) else {}
        # Ordering metadata is an infrastructure invariant and must survive domain
        # state transitions (IDLE/CLARIFYING/CHOOSING/ESCALATED). Callers should not
        # have to know about the watermark when replacing their business state.
        ordering = existing_data.get("_ordering") if isinstance(existing_data.get("_ordering"), dict) else None
        routing_memory = existing_data.get("_routing_memory") if isinstance(existing_data.get("_routing_memory"), list) else None
        new_data = dict(data) if isinstance(data, dict) else {}
        # Routing memory is conversation-scoped memory, not state-machine data.
        # Preserve it when moving IDLE/CLARIFYING/CHOOSING/REPHRASING states so a
        # later turn still knows what Terra has already answered.
        if ordering and "_ordering" not in new_data:
            new_data["_ordering"] = ordering
        if routing_memory and "_routing_memory" not in new_data:
            new_data["_routing_memory"] = routing_memory[-20:]
        current.state = state
        current.data = new_data
        await self.session.flush()

    async def append_routing_memory(self, conversation_id: str, items: list[dict[str, str]]) -> None:
        """Append memory only after the corresponding customer reply was delivered."""
        if not items:
            return
        current = await self.get(conversation_id, for_update=True)
        data = dict(current.data) if isinstance(current.data, dict) else {}
        existing = data.get("_routing_memory") if isinstance(data.get("_routing_memory"), list) else []
        clean: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kb_id = str(item.get("kb_id") or "").strip()
            answer = str(item.get("answer") or "").strip()
            question = str(item.get("question") or "").strip()
            if kb_id and answer:
                clean.append({"kb_id": kb_id, "question": question[-500:], "answer": answer[-1500:]})
        if clean:
            data["_routing_memory"] = (existing + clean)[-20:]
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


class ConversationTurnRepository:
    """Persist and retrieve a role-labelled transcript for LLM routing."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def ensure_user_turn(
        self,
        *,
        entity_id: str,
        channel: str,
        text: str,
        event_id: int | None,
        provider_part_id: str | None,
        turn_at: datetime | None = None,
        dedup_key: str | None = None,
    ) -> ConversationTurn:
        key = dedup_key or (
            f"provider:{provider_part_id}" if provider_part_id
            else f"event:{event_id}:user"
        )
        existing = await self.session.scalar(select(ConversationTurn).where(ConversationTurn.dedup_key == key))
        if existing:
            return existing
        row = ConversationTurn(
            entity_id=entity_id, channel=channel, role="user", text=str(text or "").strip(),
            status="sent", event_id=event_id, provider_part_id=provider_part_id,
            dedup_key=key, turn_at=turn_at or datetime.now(timezone.utc),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def ensure_assistant_turn(
        self,
        *,
        entity_id: str,
        channel: str,
        text: str,
        event_id: int | None,
        action_key: str,
        status: str = "pending",
    ) -> ConversationTurn:
        key = f"assistant:{action_key}"
        existing = await self.session.scalar(select(ConversationTurn).where(ConversationTurn.dedup_key == key))
        if existing:
            return existing
        row = ConversationTurn(
            entity_id=entity_id, channel=channel, role="assistant", text=str(text or "").strip(),
            status=status, event_id=event_id, action_key=action_key, dedup_key=key,
            turn_at=datetime.now(timezone.utc),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def import_context(
        self,
        *,
        entity_id: str,
        channel: str,
        context: list[dict[str, str]],
        current_turn_at: datetime | None = None,
    ) -> None:
        """Seed the local transcript from provider history on first use.

        Provider context is already chronological. Synthetic timestamps keep that
        ordering while marking records as imported so they never masquerade as
        fresh webhook events.
        """
        base = current_turn_at or datetime.now(timezone.utc)
        total = len(context or [])
        previous_signature = None
        for idx, item in enumerate(context or []):
            role = str(item.get("role") or "").strip().lower()
            text = str(item.get("text") or "").strip()
            if role not in {"user", "assistant"} or not text:
                continue
            signature = (role, text)
            if signature == previous_signature:
                continue
            previous_signature = signature
            synthetic_at = base - timedelta(seconds=max(1, total - idx))
            # Keep the import key stable across repeated provider fetches. The old
            # key included the list index, which meant a later fetch containing
            # one additional older message could generate a different key for every
            # previously imported turn. That made retries capable of duplicating the
            # local transcript. Exact duplicate role+text turns are intentionally
            # collapsed here; provider history remains an enrichment source, while
            # webhook-authored current turns are persisted separately with event IDs.
            import_hash = __import__("hashlib").sha256(
                f"{entity_id}|{channel}|{role}|{text}".encode("utf-8")
            ).hexdigest()
            key = f"import:{import_hash}"
            existing = await self.session.scalar(select(ConversationTurn).where(ConversationTurn.dedup_key == key))
            if existing:
                continue
            self.session.add(ConversationTurn(
                entity_id=entity_id, channel=channel, role=role, text=text,
                status="sent", dedup_key=key, turn_at=synthetic_at,
            ))
        await self.session.flush()

    async def count_user_webhook_turns(self, entity_id: str, *, exclude_event_id: int | None = None) -> int:
        """Count persisted customer webhook turns, excluding the current event.

        This is the authoritative first-turn signal for the worker. Counting inbound
        events is race-prone because several webhooks can already be queued before the
        first one is processed. Imported provider history is deliberately excluded.
        """
        conditions = [
            ConversationTurn.entity_id == entity_id,
            ConversationTurn.role == "user",
            ConversationTurn.event_id.is_not(None),
        ]
        if exclude_event_id is not None:
            conditions.append(ConversationTurn.event_id != int(exclude_event_id))
        result = await self.session.scalar(
            select(func.count(ConversationTurn.id)).where(*conditions)
        )
        return int(result or 0)

    async def count(self, entity_id: str) -> int:
        result = await self.session.scalar(
            select(func.count(ConversationTurn.id)).where(ConversationTurn.entity_id == entity_id)
        )
        return int(result or 0)

    async def recent_context(self, entity_id: str, *, limit: int = 24) -> list[dict[str, str]]:
        rows = (await self.session.execute(
            select(ConversationTurn)
            .where(
                (ConversationTurn.entity_id == entity_id)
                & (ConversationTurn.status == "sent")
            )
            .order_by(desc(ConversationTurn.turn_at), desc(ConversationTurn.id))
            .limit(max(1, int(limit)))
        )).scalars().all()
        rows.reverse()
        return [{"role": row.role, "text": row.text} for row in rows]

    async def mark_action_status(self, action_key: str, status: str) -> None:
        await self.session.execute(
            update(ConversationTurn)
            .where(ConversationTurn.action_key == action_key)
            .values(status=status)
        )
        await self.session.flush()
