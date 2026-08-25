from datetime import datetime
from typing import Any
from sqlalchemy import DateTime, Integer, String, Text, JSON, Enum as SqlEnum, Boolean, Float, func, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database.base import Base
from app.domain.enums import EventStatus, ConversationStateKind


class InboundEvent(Base):
    __tablename__ = "inbound_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    provider_event_id: Mapped[str] = mapped_column(String(128), index=True)
    topic: Mapped[str] = mapped_column(String(128), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[EventStatus] = mapped_column(SqlEnum(EventStatus, name="event_status"), default=EventStatus.NEW, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationState(Base):
    __tablename__ = "conversation_states"
    conversation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    state: Mapped[ConversationStateKind] = mapped_column(
        SqlEnum(ConversationStateKind, name="conversation_state"),
        default=ConversationStateKind.IDLE,
        nullable=False,
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EscalatedConversation(Base):
    """Persistent operator hand-off blocklist.

    One active row exists per Intercom conversation/ticket. As soon as an entity
    is escalated, future inbound events are ignored before the decision pipeline
    is invoked, so the bot cannot compete with a human operator.
    """

    __tablename__ = "escalated_conversations"

    conversation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    user_email: Mapped[str] = mapped_column(String(320), default="", nullable=False)
    last_message_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    last_message_at: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    source_event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    escalated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_escalated_conversations_channel_reason", "channel", "reason"),
    )


class SupportInteraction(Base):
    __tablename__ = "support_interactions"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    channel: Mapped[str] = mapped_column(String(32))
    question: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(64), index=True)
    kb_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    query_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    semantic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lexical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fuzzy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_intent_evidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    intent_fit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    family_fit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_support_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ontology_alignment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    negative_contrast_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    contrastive_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    contrastive_competitor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    contrastive_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    automated: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
