from dataclasses import dataclass, field
from typing import Any
from app.domain.enums import Channel, Decision, EscalationReason

@dataclass(slots=True)
class InboundMessage:
    event_id: str
    event_key: str
    topic: str
    entity_id: str
    channel: Channel
    text: str
    user_name: str = ""
    user_email: str = ""
    updated_at: str = ""
    has_attachments: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class DecisionTrace:
    decision: Decision
    question: str
    kb_id: str | None = None
    category: str | None = None
    score: float = 0.0
    margin: float = 0.0
    embedding_score: float = 0.0
    intent_fit_score: float = 0.0
    family_fit_score: float = 0.0
    query_quality_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    fuzzy_score: float = 0.0
    evidence_score: float = 0.0
    real_intent_evidence: float = 0.0
    domain_support_score: float = 0.0
    ontology_alignment_score: float = 0.0
    negative_contrast_score: float = 0.0
    decision_reason: str = ""
    escalation_reason: EscalationReason | None = None
    llm_used: bool = False
    automated: bool = False
    contrastive_margin: float = 0.0
    contrastive_competitor_id: str | None = None
    contrastive_reason: str = ""

@dataclass(slots=True)
class OutboundAction:
    kind: str
    text: str | None = None
    escalate: bool = False

@dataclass(slots=True)
class ProcessingResult:
    traces: list[DecisionTrace] = field(default_factory=list)
    actions: list[OutboundAction] = field(default_factory=list)
    latency_ms: float = 0.0

