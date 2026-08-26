from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True, slots=True)
class QueryUnderstanding:
    """Structured meaning extracted from a user message.

    This object intentionally describes the query, not the selected intent.
    Unknown values are represented by empty sets instead of being guessed from
    the closest KB candidate.
    """

    domain: FrozenSet[str] = frozenset()
    objects: FrozenSet[str] = frozenset()
    actions: FrozenSet[str] = frozenset()
    states: FrozenSet[str] = frozenset()
    facets: FrozenSet[str] = frozenset()
    negated: bool = False
    specificity: str = "low"
    signals: FrozenSet[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ContractEvidence:
    """Deterministic evidence result for an intent contract."""

    status: str  # proven / insufficient / contradicted / unknown
    reason: str
    required_objects: FrozenSet[str] = frozenset()
    matched_objects: FrozenSet[str] = frozenset()
    required_facets: FrozenSet[str] = frozenset()
    matched_facets: FrozenSet[str] = frozenset()
    missing: FrozenSet[str] = frozenset()
    contradictions: FrozenSet[str] = frozenset()
    strength: float = 0.0


# Object vocabulary is deliberately narrow. Generic words such as
# "верификация", "код" or "проблема" are not object evidence by themselves.
OBJECT_TERMS: dict[str, tuple[str, ...]] = {
    "phone": (
        "телефон", "номера", "номер", "смс", "sms", "телефону",
        "телефона", "телефоном", "номер телефона",
    ),
    "income": (
        "доход", "дохода", "доходов", "заработок", "заработка",
        "уровень дохода", "подтверждение дохода",
    ),
    "membership": (
        "клуб", "членство", "участник клуба", "участие в клубе",
    ),
    "subscription": ("подписка", "подписку", "подписки"),
    "rent": ("аренда", "аренду", "арендной"),
    "qr_code": ("qr", "qr-код", "qr код", "куар"),
    "invite_code": ("инвайт", "инвайт-код", "код приглашения"),
    "telegram_chat": ("телеграм", "telegram", "чат", "группа"),
    "app": ("приложение", "приложении", "приложения"),
    "school": ("школа", "школу", "школе", "школой", "школы"),
    "training": ("обучение", "учиться", "учебу", "учеба", "обучения"),
    "mentor": ("наставник", "наставника", "ментор", "менторство", "наставничество"),
    "flow": ("поток", "потока", "потоке", "запуск", "набора"),
    "account": ("аккаунт", "учетная запись", "учётная запись"),
}

OBJECT_TO_DOMAIN: dict[str, str] = {
    "phone": "account",
    "income": "income",
    "membership": "membership",
    "subscription": "payments",
    "rent": "payments",
    "qr_code": "codes",
    "invite_code": "codes",
    "telegram_chat": "telegram",
    "app": "app",
    "school": "education",
    "training": "education",
    "mentor": "mentor",
    "flow": "education",
    "account": "account",
}

# Intents that are unsafe to infer from a generic topic word. The query must
# identify the defining object (and for some intents, the defining facet).
REQUIRED_OBJECTS: dict[str, FrozenSet[str]] = {
    # High-risk semantic neighbours: generic topic words must never establish
    # these intents. Object evidence is a true hard precondition.
    "phone_verification": frozenset({"phone"}),
    "qr_code": frozenset({"qr_code"}),
    "invite_code": frozenset({"invite_code"}),
    "technical_failure": frozenset({"app"}),
    "education_price": frozenset({"school", "training"}),
    # Mentorship role split: mentee vs mentor must not be confused.
    "become_mentee": frozenset({"mentor"}),
    "online_mentorship": frozenset({"mentor"}),
}


# These generic facets are insufficient to establish an object on their own.
GENERIC_FACETS = frozenset({"verification", "technical_failure", "access", "registration"})


def _contains_any(normalized: str, terms: tuple[str, ...]) -> bool:
    return any(term in normalized for term in terms)


def understand_query(
    normalized: str,
    *,
    facets: set[str] | frozenset[str] = frozenset(),
    ontology: dict[str, set[str]] | None = None,
) -> QueryUnderstanding:
    """Build structured query meaning without consulting retrieval scores."""

    q = normalized.lower().replace("ё", "е")
    objects: set[str] = set()
    signals: set[str] = set()
    for obj, terms in OBJECT_TERMS.items():
        if _contains_any(q, terms):
            objects.add(obj)
            signals.add(f"object:{obj}")

    # Remove broad context objects when a defining object is present.
    if "phone" in objects and "account" in objects:
        objects.discard("account")
    if "school" in objects and "training" in objects:
        # Treat these as compatible alternatives; both can satisfy education
        # contracts. Keep both for explicit evidence accounting.
        pass

    actions: set[str] = set((ontology or {}).get("action", set()))
    states: set[str] = set((ontology or {}).get("state", set()))
    query_facets = set(facets)

    if "verification" in query_facets or _contains_any(q, ("верификаци", "верифиц", "подтверждени", "подтвердить")):
        query_facets.add("verification")
        signals.add("facet:verification")
    if _contains_any(q, ("не могу", "не получается", "не работает", "ошибка", "проблем", "не приходит", "не открывается")):
        states.add("failure")
        signals.add("state:failure")
    if _contains_any(q, ("хочу", "хотел", "хотела", "как стать", "стать ")):
        actions.add("intent")
    if "price" in query_facets:
        actions.add("pay")
    if "registration" in query_facets:
        actions.add("register")

    if "phone" in objects:
        query_facets.add("phone_specific")
    if "income" in objects:
        query_facets.add("income_specific")

    domain = {OBJECT_TO_DOMAIN[o] for o in objects if o in OBJECT_TO_DOMAIN}
    if not domain and ontology:
        domain = set((ontology.get("domain") or set()))

    negated = _contains_any(q, ("не хочу", "не желаю", "не собираюсь", "не буду", "не нужен", "не нужна"))
    defining_objects = objects - {"school", "training", "flow", "account", "mentor"}
    specificity = "high" if defining_objects or len(objects) >= 2 else ("medium" if objects or query_facets - GENERIC_FACETS else "low")

    return QueryUnderstanding(
        domain=frozenset(domain),
        objects=frozenset(objects),
        actions=frozenset(actions),
        states=frozenset(states),
        facets=frozenset(query_facets),
        negated=negated,
        specificity=specificity,
        signals=frozenset(signals),
    )
