from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rapidfuzz import fuzz

from app.domain.response_composer import clean_business_answer
from app.domain.intent_contracts import (
    ContractEvidence, QueryUnderstanding, REQUIRED_OBJECTS, understand_query,
)

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger("knowledge_base")


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
# Generic MMARCO reranker is optional. It is NEVER used as a hard gate.
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

RETRIEVAL_TOP_K = 20
FINAL_CANDIDATES = 8
LLM_TOP_K = 3
NEGATIVE_SEMANTIC_WEIGHT = 0.14
OBJECT_MISMATCH_PENALTY = 0.22
STATE_MISMATCH_PENALTY = 0.16
ACTION_MISMATCH_PENALTY = 0.14
FACET_MISMATCH_PENALTY = 0.12
HARD_NEGATIVE_MARGIN = 0.02

# Negative/abstention policy. These are deliberately conservative: when the
# query contains explicit structured meaning that a candidate contradicts,
# semantic similarity must not be allowed to manufacture an intent.
ONTOLOGY_ALIGNMENT_MIN = 0.60
ONTOLOGY_HARD_MISMATCH = 0.25
NEGATIVE_CONTRAST_MIN = 0.015
NEGATIVE_CONTRAST_UNSAFE = -0.015
MIN_EXPLICIT_SUPPORT_FOR_NONHARD = 0.42
MIN_DOMAIN_SUPPORT_FOR_AUTO = 0.34

# A few high-precision external-target patterns. They are intentionally narrow
# and are only used in combinations such as "стоимость обучения в <X>" or
# "курс <X>", never as a general named-entity detector.
EXTERNAL_TARGET_RE = re.compile(
    r"\b(?:в|для|у|про|на)\s+(?P<entity>[А-ЯЁ][А-ЯЁа-яё-]{2,})",
)
KNOWN_INTERNAL_ENTITIES = {
    "терра", "терре", "терру", "терры", "terra",
    # Generic modifiers/pronouns are not external entities. Without this guard,
    # phrases like «в другой школе» can be misread as an external target.
    "другой", "другая", "другое", "других", "другом", "другую",
    "своей", "свою", "своей", "нашей", "нашу", "новой", "новую",
}
EXTERNAL_TARGET_CONTEXT = (
    "обучение", "школ", "курс", "поток", "занят", "обучаться",
    "учиться", "стоимость", "цена", "записаться", "поступить",
)

# Relative/ranking signals are more trustworthy than absolute E5 cosine values.
RRF_K = 60.0

# Lexical BM25-like scorer.
BM25_K1 = 1.35
BM25_B = 0.70

# Semantic score aggregation.
SEMANTIC_MAX_WEIGHT = 0.50
SEMANTIC_TOP3_WEIGHT = 0.30
SEMANTIC_CATEGORY_WEIGHT = 0.20

# Evidence adjustments are intentionally small; ranking must remain semantic.
POSITIVE_LEXICAL_BONUS = 0.16
REQUIRED_GROUP_BONUS = 0.08
NEGATIVE_LEXICAL_PENALTY = 0.22
CONFLICT_PENALTY = 0.16
CONFLICT_COMPETITOR_MARGIN = 0.10
GENERIC_RERANK_WEIGHT = 0.10

# Decision thresholds. These are routing thresholds, NOT probabilities.
# The score is deliberately bounded to [0, 1].
# v22 STRICT: TEMPLATE is rare — only near-perfect evidence + proven contract.
# Ambiguous cases go to CATEGORY_CHOICE (Top-3) or escalate/rephrase.
# v22.1: TEMPLATE is ultra-niche. Default middle path is CATEGORY_CHOICE (Top-3).
AUTO_SCORE = 0.92
AUTO_MARGIN = 0.14
LLM_SCORE = 0.48
LLM_MARGIN = 0.030
HIGH_SCORE_LOW_MARGIN = 0.88
LEXICAL_EVIDENCE_MIN = 0.32
SEMANTIC_GAP_FOR_LLM = 0.025
MIN_REAL_INTENT_EVIDENCE = 0.36
MIN_REAL_INTENT_FIT = 0.48
MIN_TEMPLATE_EVIDENCE = 0.60

# Strong-evidence TEMPLATE override: near-duplicate / almost-certain only.
STRONG_EVIDENCE_TEMPLATE_SCORE = 0.88
STRONG_EVIDENCE_TEMPLATE_MARGIN = 0.10
STRONG_EVIDENCE_TEMPLATE_FIT = 0.78
STRONG_EVIDENCE_TEMPLATE_FAMILY = 0.50
STRONG_EVIDENCE_MIN_REAL = 0.58

# Retrieval/ranking weights. Kept bounded and re-normalized when reranker is off.
SEMANTIC_MAX_WEIGHT = 0.34
SEMANTIC_TOP3_WEIGHT = 0.16
SEMANTIC_CATEGORY_WEIGHT = 0.10
LEXICAL_SCORE_WEIGHT = 0.10
FUZZY_SCORE_WEIGHT = 0.02
INTENT_FIT_WEIGHT = 0.16
FAMILY_FIT_WEIGHT = 0.04
RRF_WEIGHT = 0.04
RERANK_WEIGHT = 0.10

# Safety thresholds: TEMPLATE ultra-strict; Top-3 deliberately permissive.
INTENT_FIT_TEMPLATE_MIN = 0.78
INTENT_FIT_LLM_MIN = 0.38
INTENT_FIT_HARD_MISMATCH = 0.25
MIN_QUERY_QUALITY_TEMPLATE = 0.32
MIN_QUERY_QUALITY_LLM = 0.18
MIN_QUERY_QUALITY_CHOICE = 0.12
MIN_FAMILY_FIT_TEMPLATE = 0.48

# User-facing fallback routing.
REPHRASE_SCORE_MAX = 0.50
REPHRASE_INTENT_FIT_MAX = 0.25
REPHRASE_QUERY_QUALITY_MAX = 0.50

# Open-set / OOS gate. It is intentionally independent of raw embedding similarity.
# A coherent message is OOS only when we have both weak supported-domain evidence
# and no defensible intent evidence. Rare in-domain questions should fall through
# to REPHRASE/CHOICE rather than being rejected by the scope gate.
OOS_TOP_SCORE_MAX = 0.56
OOS_INTENT_FIT_MAX = 0.36
OOS_REAL_EVIDENCE_MAX = 0.30
OOS_DOMAIN_SUPPORT_MAX = 0.18
OOS_QUERY_QUALITY_MIN = 0.34
OOS_EXTERNAL_TARGET = True
OOS_MIN_DOMAIN_TERM_HITS = 0

# CATEGORY_CHOICE is the primary middle mode (v22.1).
# Prefer showing 2–3 options over TEMPLATE or empty rephrase.
CATEGORY_CHOICE_MIN_SCORE = 0.42
CATEGORY_CHOICE_MIN_INTENT_FIT = 0.40
CATEGORY_CHOICE_MAX_MARGIN = 0.35
CATEGORY_CHOICE_MIN_SCORE_GAP_FROM_BEST = 0.12
CATEGORY_CHOICE_MAX_CANDIDATES = 3
CATEGORY_CHOICE_MIN_CANDIDATES = 2
CATEGORY_CHOICE_MIN_REAL_EVIDENCE = 0.28
CATEGORY_CHOICE_REQUIRE_DIRECT_SUPPORT = False

# Contrastive routing: a candidate is not considered safe merely because it
# scores well; it must also beat its declared/confusable competitors on
# intent-specific evidence. These are routing thresholds, not probabilities.
CONTRASTIVE_AUTO_MARGIN = 0.10
CONTRASTIVE_LLM_MARGIN = 0.035
CONTRASTIVE_AMBIGUITY_MARGIN = 0.055
CONTRASTIVE_MIN_COMPETITOR_SCORE = 0.50
CONTRASTIVE_MAX_CONTEXT_NEIGHBORS = 6
CONTRASTIVE_SIGNAL_BONUS = 0.16
CONTRASTIVE_NEGATIVE_BONUS = 0.18
CONTRASTIVE_ONTOLOGY_BONUS = 0.10
CONTRASTIVE_FACET_BONUS = 0.10

# Conservative ambiguity guard: a second candidate with real intent evidence
# must not be ignored merely because it is not declared in conflict_ids.
SECONDARY_EVIDENCE_MIN = 0.30
SECONDARY_EVIDENCE_MARGIN = 0.055

# Fuzzy shortcut is intentionally very strict and additionally requires
# semantic/intention agreement after the exact phrase match.
FUZZY_AUTO_MIN = 98.0
FUZZY_AUTO_MARGIN = 10.0

# Hard rejection for unusable input.
MIN_MEANINGFUL_CHARS = 3
MIN_TOKENS = 2
MIN_ALNUM_RATIO = 0.45
MAX_REPEAT_CHAR_RATIO = 0.72

# Structured negation: these operators negate a proposed user goal/choice.
# Problem-state negations such as "не могу записаться" are deliberately NOT
# treated as intent contradiction; they describe a failure state instead.
STRUCTURED_NEGATION_PREFIXES = (
    "не хочу", "не желаю", "не собираюсь", "не планирую",
    "не буду", "не готов", "не нужен", "не нужна", "не нужно",
    "не интересует", "не интересно", "не подходит", "не согласен",
)
PROBLEM_STATE_NEGATION_PREFIXES = (
    "не могу", "не получается", "не работает", "не открывается",
    "не запускается", "не приходит", "не пускает", "не дает",
    "не дает", "нет кнопки", "нет доступа",
)

MAX_SUBQUESTIONS = 5

# Critical intent contracts. These are semantic preconditions for categories
# that are easy to trigger by topic-word overlap alone. A candidate may be
# retrieved for context, but it must not become an automatic answer unless the
# query expresses the action/state that defines the intent.
INTENT_REQUIRED_FACETS: dict[str, set[str]] = {
    "personal_mentor": {"personal_mentor"},
    "mentor_application": {"mentor_application"},
    "mentor_program": {"mentor_program"},
    "mentor_application_retry": {"retry"},
    "registration_failure": {"registration"},
    "school_registration": {"registration", "school"},
    "education_entry": {"procedure"},
    "school_started": {"current_flow"},
    "flow_date": {"date"},
    "education_price": {"price"},
    "phone_verification": {"verification"},
    "telegram_link": {"telegram_link"},
    "telegram_membership": {"telegram_membership"},
    "invite_code": {"invite_code"},
    "qr_code": {"qr_code"},
}

# Some high-signal facets need morphology-aware detection instead of exact
# phrases. They are intentionally narrow to avoid turning generic words into
# false intents.

# Hard rejection for unusable input.
MIN_MEANINGFUL_CHARS = 3
MIN_ALNUM_RATIO = 0.45
MAX_REPEAT_CHAR_RATIO = 0.72

MAX_SUBQUESTIONS = 5

# Only generic function words. Negation and operational words remain.
STOP_WORDS = {
    "что", "это", "у", "меня", "мой", "моя", "мое", "моё",
    "нужно", "надо", "вы", "я", "он", "она",
    "пожалуйста", "помогите", "скажите", "подскажите", "ли",
    "а", "и", "же", "бы", "быть", "мне", "мы", "они", "как",
}

# The first layer is deterministic and deliberately conservative.
# Rules use intent-specific boundaries instead of generic phrase substring matches.
HARD_INTENT_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("kb_024", ("из всех школ", "из всех направлений", "отовсюду", "из всего обучения", "нигде не учиться"), ()),
    ("kb_006", ("сменить школу", "перейти в другую школу", "перевестись в другую школу", "сменить направление"), ()),
    ("kb_008", ("восстановиться в школе", "вернуться после отчисления", "вернуться после ухода", "хочу обратно в школу", "восстановите меня"), ("стать наставником",)),
    ("kb_007", ("исключили из школы", "выгнали из школы", "отчислили из школы", "меня исключили", "меня отчислили"), ()),
    ("kb_005", ("выйти из школы", "уйти из школы", "отчислиться из школы", "покинуть школу", "удалите меня из школы"), ("другую школу", "перейти",)),
    ("kb_023", ("хочу стать наставником", "стать наставником", "хочу быть наставником", "стать ментором", "хочу преподавать", "вести школу", "открыть свою школу", "стать спикером"), ("мой наставник",)),
    ("kb_034", ("программа наставничества", "что такое наставничество", "как работает наставничество", "расскажите про наставничество", "информация о наставничестве", "подробности о наставничестве"), ("стать наставником", "мой наставник")),
    ("kb_025", ("повторно подать заявку наставнику", "заново подать заявку наставнику", "заявка к наставнику отклонена", "наставник отказал в заявке"), ()),
    ("kb_009", ("мой наставник", "моего наставника", "мой ментор", "у меня наставник", "наставник не отвечает", "сменить наставника", "кто мой наставник", "как найти наставника"), ("стать наставником",)),
    ("kb_015", ("школа уже началась", "уже идет школа", "уже идет поток", "опоздал на поток", "опоздал на школу", "уже стартовала школа", "можно ли сейчас в идущий поток"), ()),
    ("kb_016", ("когда следующий поток", "когда новый поток", "когда следующий старт", "даты потоков", "когда следующий набор", "дата старта нового потока"), ()),
    ("kb_002", ("не могу записаться", "не получается записаться", "не дает записаться", "не пускает в школу", "нет кнопки записи", "ошибка при записи", "не могу найти кнопку записи"), ("как записаться в школу", "как попасть в школу",)),
    ("kb_033", ("как записаться в школу", "как попасть в школу", "как вступить в школу", "как записаться на поток", "как стать учеником школы", "как начать обучение"), ("не могу записаться", "ошибка при записи", "школа уже началась", "когда следующий поток")),
    ("kb_035", ("как записаться на обучение", "как попасть на обучение", "как начать обучение", "как начать обучение в терре", "хочу попасть на обучение", "хочу учиться в терре", "как попасть на поток", "как попасть на запуск", "что нужно чтобы попасть на обучение", "как стать участником обучения", "хочу учиться", "как начать учиться в терре"), ("как записаться в школу", "не могу записаться", "ошибка при записи", "школа уже началась", "когда следующий поток")),
    ("kb_004", ("ссылка на чат", "ссылка на телеграм", "ссылка в телеграм", "ссылка тг", "ссылка на группу"), ("не добавили в чат", "добавьте меня в чат")),
    ("kb_021", ("добавьте меня в чат", "не добавили в чат", "не добавили в группу", "нет в чате школы", "нет в группе школы", "добавьте меня в телеграм", "не включили в чат"), ("ссылка не открывается", "ссылка на чат не работает")),
    ("kb_013", ("инвайт-код", "код приглашения", "не работает инвайт", "инвайт не работает"), ("qr-код",)),
    ("kb_014", ("qr-код", "qr код", "не работает qr", "не сканируется qr"), ("инвайт",)),
    ("kb_018", ("повторная оплата аренды", "аренда списалась дважды", "двойная оплата аренды", "снова просят оплатить аренду"), ()),
)

# Families reduce the search space conceptually and help LLM judge with conflicts.
FAMILIES: dict[str, str] = {
    "kb_001": "app",
    "kb_002": "education",
    "kb_003": "education",
    "kb_015": "education",
    "kb_016": "education",
    "kb_033": "education",
    "kb_035": "education",
    "kb_005": "schools",
    "kb_006": "schools",
    "kb_007": "schools",
    "kb_008": "schools",
    "kb_024": "schools",
    "kb_009": "mentor",
    "kb_023": "mentor",
    "kb_025": "mentor",
    "kb_034": "mentor",
    "kb_010": "income",
    "kb_011": "payments",
    "kb_012": "payments",
    "kb_018": "payments",
    "kb_013": "codes",
    "kb_014": "codes",
    "kb_019": "access",
    "kb_017": "access",
    "kb_004": "telegram",
    "kb_021": "telegram",
    "kb_022": "support",
    "kb_026": "account",
    "kb_027": "account",
    "kb_029": "education_admin",
    "kb_030": "education_admin",
    "kb_031": "feedback",
    "kb_032": "partnership",
    "kb_020": "spaces",
}


class MatchMode(str, Enum):
    NO_MATCH = "no_match"
    TEMPLATE = "template"
    LLM = "llm"
    OPERATOR = "operator"
    CLARIFICATION = "clarification"
    CATEGORY_CHOICE = "category_choice"
    REPHRASE = "rephrase"


@dataclass(slots=True)
class MatchResult:
    mode: MatchMode
    score: float
    category: str | None
    template_answer: str | None
    kb_id: str | None
    relevant_entries: list[dict]
    entry: dict | None = None
    second_score: float = 0.0
    margin: float = 0.0
    fuzzy_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    evidence_score: float = 0.0
    real_intent_evidence: float = 0.0
    embedding_score: float = 0.0
    intent_fit_score: float = 0.0
    intent_fit_reason: str = ""
    subquestion: str | None = None
    decision_reason: str = ""
    choice_candidates: list[dict] | None = None
    query_quality_score: float = 0.0
    family_fit_score: float = 0.0
    top_candidates: list[dict] | None = None
    domain_support_score: float = 0.0
    ontology_alignment_score: float = 0.0
    negative_contrast_score: float = 0.0
    contrastive_margin: float = 0.0
    contrastive_competitor_id: str | None = None
    contrastive_reason: str = ""


@dataclass(slots=True)
class QueryQuality:
    score: float
    valid: bool
    reason: str
    tokens: int
    domain_overlap: float
    question_signal: float
    action_signal: float
    noise_signal: float
    problem_signal: float = 0.0
    meta_noise: bool = False




# ============================================================
# TEXT / QUALITY
# ============================================================


def normalize_light(text: str) -> str:
    text = (text or "").lower().strip().replace("ё", "е")
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    return " ".join(
        word for word in normalize_light(text).split()
        if word not in STOP_WORDS and len(word) > 1
    )


def tokenize(text: str) -> list[str]:
    return [w for w in normalize(text).split() if len(w) > 1]


# Lightweight Russian morphology helper used only for deterministic intent
# signals. It is deliberately conservative: semantic retrieval remains the
# main generalizer, while this catches common inflection/typo variants.
_RU_SUFFIXES = (
    "ами", "ями", "ого", "ему", "ому", "ими", "ыми", "ей", "ов", "ев",
    "ам", "ям", "ом", "ем", "ым", "им", "ах", "ях", "ами", "ями",
    "ы", "и", "а", "я", "о", "е", "у", "ю", "ь",
)

def light_stem(token: str) -> str:
    t = normalize_light(token)
    if len(t) <= 4:
        return t
    for suffix in _RU_SUFFIXES:
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[:-len(suffix)]
    return t


def stem_tokens(text: str) -> list[str]:
    return [light_stem(t) for t in normalize_light(text).split() if len(t) > 1]


def contains_operator_request(text: str) -> bool:
    q = normalize_light(text)
    return bool(re.search(
        r"\bоператор\b|соедините(?:\s+меня)?\s+с\s+оператором|"
        r"переключите\s+на\s+оператора|хочу\s+оператора|"
        r"позовите\s+оператора|нужен\s+оператор",
        q,
    ))


# High-risk intents: never auto-answer with TEMPLATE. Escalate + collect data.
# Patterns are deliberately narrow and phrase-oriented to avoid false positives.
# Patterns are intentionally specific. Prefer false negatives (normal routing)
# over false positives that escalate "прошу вернуть доступ" as a refund.
_HIGH_RISK_PATTERNS: dict[str, tuple[str, ...]] = {
    "refund": (
        "вернуть деньги",
        "вернуть денежные средства",
        "возврат оплаты",
        "возврат денег",
        "вернуть средства",
        "оплатила и не смогла",
        "оплатил и не смог",
        "деньги списали",
        "списали деньги",
        "вернуть оплату",
        "хочу возврат денег",
        "хочу возврат средств",
        "прошу вернуть деньги",
        "прошу вернуть средства",
        "прошу вернуть оплату",
        "просьба вернуть деньги",
    ),
    "personal_data": (
        "удалить персональные данные",
        "удалить мои данные",
        "удалить из базы",
        "удалите мои данные",
        "удалите персональные",
        "отзываю согласие",
        "отзыв согласия",
        "удалить мои персональные",
        "удаление персональных данных",
    ),
    "complaint": (
        "жалоба на",
        "подаю жалобу",
        "хочу пожаловаться",
        "хамоватая",
        "хамит",
        "оскорбил",
        "оскорбила",
        "примите меры и удалить",
        "прошу принять меры",
    ),
}


def detect_high_risk(text: str) -> str | None:
    """Return high-risk kind (refund / personal_data / complaint) or None.

    These requests must escalate with structured data collection and must never
    receive an automatic TEMPLATE answer.
    """
    q = normalize_light(text or "")
    if not q:
        return None
    for kind, phrases in _HIGH_RISK_PATTERNS.items():
        for phrase in phrases:
            if phrase in q:
                return kind
    return None


def is_non_query_message(text: str) -> bool:
    """Exact generic acknowledgements/greetings/noise only."""
    q = normalize_light(text)
    if not q:
        return True
    return q in _NON_QUERY_EXACT or q in META_NOISE_PATTERNS


def _quality_gate(text: str) -> tuple[bool, str]:
    quality = _query_quality(text)
    return quality.valid, quality.reason


META_NOISE_PATTERNS: tuple[str, ...] = (
    "бесполезное сообщение",
    "не несущее смысла",
    "не имеет смысла",
    "просто сообщение",
    "бессмысленное сообщение",
    "это бесполезно",
    "просто набор слов",
    "набор бессмысленных",
)

# Exact acknowledgements/greetings are ignored as non-support messages.
_NON_QUERY_EXACT: set[str] = set()
try:
    _non_query_path = Path(__file__).resolve().parents[3] / "data" / "non_query_phrases.json"
    _non_query_data = json.loads(_non_query_path.read_text(encoding="utf-8"))
    _NON_QUERY_EXACT = {
        normalize_light(str(x))
        for x in (_non_query_data.get("exact_phrases", []) if isinstance(_non_query_data, dict) else [])
        if isinstance(x, str) and normalize_light(x)
    }
except Exception:
    logger.warning("Could not load data/non_query_phrases.json; using built-in non-query patterns only")


QUESTION_WORDS = {
    "когда", "как", "где", "почему", "зачем", "сколько", "можно", "нельзя",
    "есть", "ли", "какой", "какая", "какие", "кто", "куда", "откуда",
}

ACTION_PREFIXES = (
    "можн", "хоч", "нуж", "получ", "попаст", "запис", "перейд", "перевест",
    "смен", "выйт", "уйт", "восстанов", "верн", "добав", "отк",
    "оплат", "плат", "запуст", "старт", "нача", "учит", "работ", "войд",
    "получит", "найт", "провер", "исправ", "удал", "отмен", "подат",
)

DOMAIN_TERMS = {
    # Generic domain nouns and support concepts. This is intentionally broad;
    # category-specific fit is still computed separately.
    "школ", "обуч", "учеб", "поток", "настав", "ментор", "чат", "телеграм",
    "тг", "инвайт", "qr", "код", "аккаунт", "профил", "регистрац", "прилож",
    "доступ", "оплат", "платеж", "аренд", "деньг", "возврат", "выплат",
    "мероприят", "курс", "запис", "заявк", "ссылк", "групп", "телефон",
    "номер", "смс", "парол", "почт", "email", "почта",
}


def _token_prefix_hits(tokens: list[str], prefixes: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if token.startswith(prefixes))


def _query_quality(text: str, domain_terms: set[str] | None = None) -> QueryQuality:
    q = normalize_light(text)
    if not q:
        return QueryQuality(0.0, False, "empty", 0, 0.0, 0.0, 0.0, 1.0, 0.0)

    compact = q.replace(" ", "")
    tokens = tokenize(q)
    token_count = len(tokens)

    # Exact generic replies/greetings remain ignorable even when one token.
    if q in _NON_QUERY_EXACT or q in META_NOISE_PATTERNS:
        return QueryQuality(
            0.0, False, "meta_noise", token_count, 0.0, 0.0, 0.0, 1.0,
            problem_signal=0.0, meta_noise=True,
        )

    if len(q) < MIN_MEANINGFUL_CHARS:
        return QueryQuality(0.0, False, "too_short", token_count, 0.0, 0.0, 0.0, 1.0, 0.0)

    if token_count < MIN_TOKENS:
        return QueryQuality(0.0, False, "single_word", token_count, 0.0, 0.0, 0.0, 1.0, 0.0)

    alnum = sum(ch.isalnum() for ch in q)
    if alnum / max(1, len(q.replace(" ", ""))) < MIN_ALNUM_RATIO:
        return QueryQuality(0.0, False, "low_alnum_ratio", token_count, 0.0, 0.0, 0.0, 1.0, 0.0)

    if compact:
        repeats = sum(1 for a, b in zip(compact, compact[1:]) if a == b)
        if repeats / max(1, len(compact) - 1) > MAX_REPEAT_CHAR_RATIO:
            return QueryQuality(0.0, False, "repeated_characters", token_count, 0.0, 0.0, 0.0, 1.0, 0.0)

    for pattern in META_NOISE_PATTERNS:
        if pattern in q:
            return QueryQuality(
                0.0, False, "meta_noise", token_count, 0.0, 0.0, 0.0, 1.0,
                problem_signal=0.0, meta_noise=True,
            )

    # Reject obvious keyboard/noise patterns such as "q q q ahajaj".
    # Use light-normalized tokens here so one-character repeated junk is not
    # removed by the normal support-tokenizer before the noise check.
    raw_tokens = q.split()
    latin_tokens = [t for t in raw_tokens if re.fullmatch(r"[a-z0-9_-]+", t)]
    repeated_token_ratio = (
        1.0 - (len(set(latin_tokens)) / len(latin_tokens))
        if latin_tokens
        else 0.0
    )
    max_repeated_token_ratio = (
        max((latin_tokens.count(token) for token in set(latin_tokens)), default=0)
        / max(1, len(latin_tokens))
    )
    if (
        len(latin_tokens) >= 3
        and (repeated_token_ratio >= 0.50 or max_repeated_token_ratio >= 0.50)
    ):
        return QueryQuality(
            0.0, False, "repeated_latin_tokens", token_count, 0.0, 0.0, 0.0, 1.0,
            problem_signal=0.0,
        )

    # Heuristic domain overlap. Category-specific vocabulary is layered on top.
    vocab = domain_terms or DOMAIN_TERMS
    domain_hits = 0
    for token in tokens:
        if any(token.startswith(prefix) for prefix in vocab):
            domain_hits += 1
    # One explicit business-domain token is already meaningful evidence;
    # additional domain tokens increase confidence.
    domain_overlap = min(1.0, 0.35 * domain_hits + 0.15 * max(0, domain_hits - 1))

    question_hits = sum(1 for token in tokens if token in QUESTION_WORDS)
    punctuation_signal = 1.0 if "?" in (text or "") else 0.0
    question_signal = min(1.0, 0.35 * question_hits + 0.45 * punctuation_signal)

    action_hits = _token_prefix_hits(tokens, ACTION_PREFIXES)
    action_signal = min(1.0, action_hits / 2.0)

    problem_signal = 1.0 if (
        any(token in {"не", "нет", "нельзя"} for token in tokens)
        and domain_overlap > 0.0
    ) else 0.0

    # A message with a domain term, an action, an explicit question form,
    # or a concrete negative/problem statement is generally meaningful.
    # generally meaningful. A bare generic sentence without any domain evidence
    # is intentionally kept below the safe thresholds.
    score = (
        0.42 * domain_overlap
        + 0.20 * question_signal
        + 0.23 * action_signal
        + 0.10 * problem_signal
        + 0.05 * min(1.0, token_count / 8.0)
    )

    valid = (
        score >= 0.24
        or domain_overlap >= 0.50
        or question_signal >= 0.55
        or action_signal >= 0.50
        or problem_signal >= 0.50
        or (domain_overlap >= 0.25 and question_signal >= 0.30)
    )
    reason = (
        "domain_and_intent_signal"
        if valid
        else "insufficient_domain_or_intent_signal"
    )
    return QueryQuality(
        score=float(min(1.0, score)),
        valid=valid,
        reason=reason,
        tokens=token_count,
        domain_overlap=float(domain_overlap),
        question_signal=float(question_signal),
        action_signal=float(action_signal),
        noise_signal=float(1.0 - score),
        problem_signal=float(problem_signal),
    )


# Pure greetings / politeness — never a standalone intent.
_GREETING_ONLY_RE = re.compile(
    r"^(?:"
    r"добрый\s+(?:день|вечер|утро)|доброй\s+ночи|доброго\s+дня|"
    r"здравствуйте|здравствуй|привет(?:ик)?|хай|здорово|"
    r"hello|hi|good\s+(?:morning|afternoon|evening)|"
    r"спасибо|благодарю|пожалуйста"
    r")\s*$",
    re.IGNORECASE,
)


def _is_greeting_only(text: str) -> bool:
    """True for standalone greetings like 'Добрый день!' with no real question."""
    q = normalize_light(text)
    if not q:
        return True
    if _GREETING_ONLY_RE.match(q):
        return True
    tokens = tokenize(q)
    # Extremely short politeness with no domain signal.
    if len(tokens) <= 2 and not any(
        t.startswith((
            "школ", "запис", "поток", "настав", "код", "чат", "оплат",
            "прилож", "смс", "телефон", "обуч", "аккаунт",
        ))
        for t in tokens
    ):
        if any(t.startswith((
            "добр", "здравств", "привет", "спасиб", "благодар",
        )) for t in tokens):
            return True
    return False


def _split_structural_questions(text: str) -> list[str]:
    chunks = re.split(r"(?:\?+|!+|\n+)", (text or "").strip())
    result: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", chunk).strip()
        if not chunk:
            continue
        for item in re.split(r"\s+(?=\d+[.)]\s+)", chunk):
            item = re.sub(r"^\s*\d+[.)]\s*", "", item).strip()
            item = re.sub(r"^(?:и|а|также)\s+", "", item, flags=re.IGNORECASE)
            if item:
                result.append(item)
    # Drop pure greetings so "Добрый день! Как записаться..." stays one intent.
    meaningful = [
        x for x in result
        if len(tokenize(x)) >= 2 and not _is_greeting_only(x)
    ]
    return meaningful[:MAX_SUBQUESTIONS] if len(meaningful) > 1 else (
        meaningful if meaningful else [text]
    )


def _split_conjunctive_question(text: str) -> list[str]:
    """Generate conservative conjunction split candidates.

    We do not assume every "и" means two intents. The caller validates the
    fragments against the actual classifier before accepting the split.
    """
    q = text.strip()
    patterns = (
        r"\s+а\s+также\s+",
        r"\s+и\s+еще\s+",
        r"\s+также\s+",
        r"\s+и\s+(?=(?:как|где|когда|можно|хочу|хотел|хотела|что|кто|почему|зачем)\b)",
        r"\s+а\s+(?=(?:как|где|когда|можно|хочу|хотел|хотела|что|кто|почему|зачем)\b)",
    )
    for pattern in patterns:
        parts = [
            p.strip(" ,;")
            for p in re.split(pattern, q, maxsplit=1, flags=re.IGNORECASE)
        ]
        parts = [
            re.sub(r"^(?:и|а|также)\s+", "", p, flags=re.IGNORECASE).strip(" ,;")
            for p in parts
        ]
        if len(parts) == 2 and all(len(normalize_light(p).split()) >= 2 for p in parts):
            return parts
    return [q]


def split_user_questions(text: str, max_questions: int = MAX_SUBQUESTIONS) -> list[str]:
    """Split only on high-confidence structural boundaries.

    Conjunctions are validated later against actual intent predictions, so a
    normal single intent containing "и" is not accidentally fragmented.
    Pure greetings are never treated as separate questions.
    """
    text = (text or "").strip()
    if not text:
        return [text]

    structural = _split_structural_questions(text)
    structural = [s for s in structural if not _is_greeting_only(s)]
    if len(structural) > 1:
        return structural[:max_questions]
    if len(structural) == 1:
        return structural
    # Everything was greeting-only — keep original for quality gate.
    return [text]


# ============================================================
# SIMPLE BM25 / RRF
# ============================================================


class BM25Index:
    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.doc_freq: dict[str, int] = {}
        self.doc_len = np.asarray([len(doc) for doc in docs], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if docs else 0.0
        for doc in docs:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self.n = len(docs)

    def score(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        if not query or not self.n:
            return scores
        q_terms = set(query)
        for i, doc in enumerate(self.docs):
            if not doc:
                continue
            tf: dict[str, int] = {}
            for term in doc:
                if term in q_terms:
                    tf[term] = tf.get(term, 0) + 1
            dl = float(self.doc_len[i])
            for term, freq in tf.items():
                df = self.doc_freq.get(term, 0)
                idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
                denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / max(self.avgdl, 1e-9))
                scores[i] += idf * (freq * (BM25_K1 + 1.0)) / denom
        return scores


def reciprocal_rank_fusion(rank_lists: Iterable[list[int]], k: float = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


# ============================================================
# KNOWLEDGE BASE
# ============================================================


class KnowledgeBase:
    """Production-oriented hybrid intent classifier.

    Stages:
      1. Input-quality / operator / deterministic hard intent rules.
      2. Hybrid candidate generation: lexical BM25 + E5 embeddings.
      3. Per-intent evidence: positive phrases, negative phrases, constraints.
      4. Optional generic CrossEncoder as a *small* ranking feature only.
      5. Intent-fit safety layer and routing.
      6. Future user-state helpers: CATEGORY_CHOICE / REPHRASE.

    Important: E5 cosine and generic CrossEncoder scores are ranking features,
    not probabilities. The classifier therefore relies heavily on relative
    ranking, lexical evidence, contradictions and margins.
    """

    def __init__(
        self,
        path: str = "data/knowledge_base.json",
        model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        device: str = "cpu",
        use_reranker: bool = False,
    ) -> None:
        self.entries: list[dict] = []
        self.entries_by_id: dict[str, dict] = {}
        self.phrase_records: list[dict[str, Any]] = []
        self.phrase_embeddings_matrix = np.empty((0, 0), dtype=np.float32)
        self.category_embeddings: dict[str, np.ndarray] = {}
        self.category_phrase_indices: dict[str, list[int]] = {}
        self.negative_phrase_records: list[dict[str, Any]] = []
        self.negative_embeddings_matrix = np.empty((0, 0), dtype=np.float32)
        self.category_negative_indices: dict[str, list[int]] = {}
        self.family_vocabulary: dict[str, set[str]] = {}
        self.default_answer = "Уточните, пожалуйста, подробнее ваш вопрос."
        self.model = None
        self.reranker = None

        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for production inference")

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)

        if use_reranker:
            if CrossEncoder is None:
                raise RuntimeError("sentence-transformers CrossEncoder is unavailable")
            logger.info("Loading OPTIONAL generic reranker: %s", reranker_model_name)
            self.reranker = CrossEncoder(reranker_model_name, device=device, max_length=512)

        self._load(path)
        self._build_indexes()

        logger.info(
            "KB v12 loaded | categories=%d | phrase_records=%d | reranker=%s",
            len(self.entries), len(self.phrase_records), self.reranker is not None,
        )

    # --------------------------------------------------------
    # LOAD / INDEX
    # --------------------------------------------------------

    def _load(self, path: str) -> None:
        kb_path = Path(path)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        data = json.loads(kb_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Knowledge base root must be a JSON list")

        for raw in data:
            if not isinstance(raw, dict):
                continue
            entry_id = str(raw.get("id") or "").strip()
            if not entry_id:
                continue
            if entry_id == "default":
                self.default_answer = clean_business_answer(raw.get("template_answer") or self.default_answer)
                continue
            if entry_id in self.entries_by_id:
                raise ValueError(f"Duplicate knowledge-base id: {entry_id}")

            entry = dict(raw)
            for field in (
                "match_phrases", "negative_phrases", "required_phrases",
                "when_to_use", "when_not_to_use", "confused_with",
                "positive_examples", "hard_negative_examples",
            ):
                value = entry.get(field, [])
                entry[field] = value if isinstance(value, list) else []

            entry["id"] = entry_id
            entry["family"] = entry.get("family") or FAMILIES.get(entry_id, "other")
            entry["priority"] = int(entry.get("priority") or 0)
            entry["template_answer"] = clean_business_answer(entry.get("template_answer"))
            entry["choice_label"] = (
                str(entry.get("choice_label") or "").strip()
                or str(entry.get("category") or entry_id).strip()
            )
            entry["constraints"] = entry.get("constraints") if isinstance(entry.get("constraints"), dict) else {}
            entry["ontology"] = self._derive_ontology(entry)

            # Keep the original phrases as the canonical retrieval examples.
            positives = entry.get("positive_examples") or entry.get("match_phrases")
            clean: list[str] = []
            seen: set[str] = set()
            for phrase in positives:
                if not isinstance(phrase, str):
                    continue
                phrase = phrase.strip()
                key = normalize_light(phrase)
                if key and key not in seen:
                    clean.append(phrase)
                    seen.add(key)
            entry["positive_examples"] = clean
            entry["match_phrases"] = clean

            entry["_negative_norm"] = [normalize_light(p) for p in entry["negative_phrases"] if isinstance(p, str)]
            entry["_positive_norm"] = [normalize_light(p) for p in entry["positive_examples"] if isinstance(p, str)]
            self.entries.append(entry)
            self.entries_by_id[entry_id] = entry

        # Build hard negatives automatically from the KB's own curated
        # conflict graph. This gives us supervised-looking contrastive evidence
        # even before a large production dataset exists. Only examples from
        # explicitly confused intents are used, so unrelated categories cannot
        # contaminate the negative channel.
        for entry in self.entries:
            existing = {normalize_light(x) for x in entry.get("hard_negative_examples", []) if isinstance(x, str)}
            conflict_ids: list[str] = [str(x) for x in entry.get("conflict_ids", []) if str(x)]
            confused = entry.get("confused_with", []) or []
            for item in confused:
                if isinstance(item, str):
                    conflict_ids.append(item)
                elif isinstance(item, dict) and item.get("kb_id"):
                    conflict_ids.append(str(item["kb_id"]))
            for conflict_id in dict.fromkeys(conflict_ids):
                other = self.entries_by_id.get(conflict_id)
                if not other:
                    continue
                # Keep the set deliberately compact: a few high-quality hard
                # negatives per conflict are more useful than flooding the
                # semantic negative index with every example in the KB.
                for phrase in other.get("positive_examples", [])[:5]:
                    key = normalize_light(phrase)
                    if key and key not in existing and key not in set(entry.get("_positive_norm", [])):
                        entry.setdefault("hard_negative_examples", []).append(phrase)
                        existing.add(key)

        self._validate_entries()

    @staticmethod
    def _derive_ontology(entry: dict) -> dict[str, str | None]:
        """Derive a stable intent signature from the KB metadata already present."""
        it = str(entry.get("intent_type") or "").strip()
        family = str(entry.get("family") or "other").strip()
        facets = {str(x).strip() for x in entry.get("intent_facets", []) if str(x).strip()}
        mapping = {
            "technical_failure": ("app", "use", "failure"),
            "registration_failure": ("school", "register", "failure"),
            "school_registration": ("school", "register", "procedure"),
            "education_entry": ("education", "enter", "procedure"),
            "missed_event": ("event", "attend", "missed"),
            "flow_date": ("flow", "ask", "information"),
            "school_started": ("school", "enter", "already_started"),
            "personal_mentor": ("mentor", "contact", "current"),
            "mentor_application": ("mentor_role", "become", "intent"),
            "mentor_program": ("mentor_program", "learn", "information"),
            "mentor_application_retry": ("mentor_application", "resubmit", "rejected"),
            "education_price": ("training", "pay", "information"),
            "subscription_charge": ("subscription", "pay", "charged"),
            "rent_payment_issue": ("rent", "pay", "charged"),
            "account_identity": ("account", "verify", "mismatch"),
            "phone_verification": ("phone", "verify", "verification"),
            "telegram_link": ("telegram_chat", "get", "information"),
            "telegram_membership": ("telegram_chat", "join", "failure"),
            "invite_code": ("invite_code", "use", "failure"),
            "qr_code": ("qr_code", "use", "failure"),
            "access": ("school", "access", "failure"),
            "voluntary_exit": ("school", "leave", "intent"),
            "school_transfer": ("school", "transfer", "intent"),
            "school_restoration": ("school", "restore", "intent"),
            "involuntary_exclusion": ("school", "restore", "excluded"),
            "global_exit": ("schools", "leave", "intent"),
        }
        obj, action, state = mapping.get(it, (family, None, None))
        facet = next(iter(sorted(facets)), None)
        for preferred in ("price", "date", "duration", "age", "registration", "technical_failure", "verification", "access"):
            if preferred in facets:
                facet = preferred
                break
        return {"domain": family, "object": obj, "action": action, "state": state, "facet": facet}

    @staticmethod
    def _query_ontology(question: str) -> dict[str, set[str]]:
        q = normalize_light(question)
        out = {"domain": set(), "object": set(), "action": set(), "state": set(), "facet": set()}
        patterns = {
            "price": ("сколько стоит", "стоимость", "цена", "платно", "платное", "платить", "бесплатно", "обойдется", "обойдётся"),
            "date": ("когда следующий", "когда новый", "когда старт", "дата старта", "даты потока", "в какой день"),
            "duration": ("сколько длится", "как долго", "сколько месяцев", "сколько недель", "длительность", "продолжительность"),
            "age": ("с какого возраста", "со скольки лет", "до какого возраста", "возраст"),
            "registration": ("записаться", "запись", "регистрация", "зарегистрироваться"),
            "technical_failure": ("не работает", "не получается", "не могу", "ошибка", "не открывается", "не принимает", "не пускает"),
            "verification": ("код подтверждения", "подтвердить номер", "не приходит смс"),
            "access": ("нет доступа", "не вижу", "не пускает", "доступ"),
            "mentor_application": ("стать наставником", "стать ментором", "хочу преподавать", "хочу быть наставником"),
            "mentor_program": ("программа наставничества", "что такое наставничество", "как работает наставничество"),
            "personal_mentor": ("мой наставник", "моего наставника", "кто мой наставник"),
            "membership": ("вступить в клуб", "вступление в клуб", "членство", "членство в клубе", "участник клуба", "участие в клубе", "стать участником клуба", "клуб"),
            "training": ("обучение", "учиться", "школа", "поток", "запуск"),
            "telegram_chat": ("телеграм", "чат", "группа"),
            "mentor": ("наставник", "ментор", "наставничество"),
            "app": ("приложение", "приложении"),
            "invite_code": ("инвайт", "код приглашения"),
            "qr_code": ("qr", "qr код", "qr-код"),
            "rent": ("аренда", "аренду"),
            "subscription": ("подписка", "подписку"),
        }
        stems = set(stem_tokens(question))
        if stems & {"наставник", "ментор"} and (stems & {"хоч", "хочу", "жел", "готов", "мог"}) and "стат" in stems:
            out["object"].add("mentor_application")
            out["action"].add("become")
            out["state"].add("intent")
        elif stems & {"наставник", "ментор"} and "будущ" in stems and (stems & {"хоч", "хочу", "жел", "готов"}):
            out["object"].add("mentor_application")
            out["action"].add("become")
            out["state"].add("intent")

        for key, vals in patterns.items():
            if any(v in q for v in vals):
                target = "facet" if key in {"price", "date", "duration", "age", "registration", "technical_failure", "verification", "access"} else "object"
                out[target].add(key)
        if out["object"]:
            out["domain"].add("known")
        if "technical_failure" in out["facet"]:
            out["state"].add("failure")
        if "registration" in out["facet"]:
            out["action"].add("register")
        if "mentor_application" in out["object"]:
            out["action"].add("become")
            out["state"].add("intent")
        if "mentor_program" in out["object"]:
            out["action"].add("learn")
            out["state"].add("information")
        if "personal_mentor" in out["object"]:
            out["action"].add("contact")
            out["state"].add("current")
        if any(v in out["object"] for v in {"subscription", "rent"}):
            out["action"].add("pay")
        if "membership" in out["object"]:
            out["action"].add("join")
            out["state"].add("information")
        if "training" in out["object"] and "price" in out["facet"]:
            out["action"].add("pay")
            out["state"].add("information")
        return out

    def _validate_entries(self) -> None:
        """Fail fast on KB structure errors that would make ranking unreliable."""
        known_ids = set(self.entries_by_id)
        for entry in self.entries:
            if not isinstance(entry.get("choice_label"), str) or not entry["choice_label"].strip():
                raise ValueError(f"KB {entry['id']}: choice_label must be a non-empty string")
            for field in ("positive_examples", "negative_phrases", "required_phrases"):
                values = entry.get(field, [])
                if not all(isinstance(x, str) and x.strip() for x in values):
                    raise ValueError(f"KB {entry['id']}: field {field} must contain non-empty strings")
            for conflict_id in entry.get("conflict_ids", []):
                if str(conflict_id) not in known_ids:
                    raise ValueError(
                        f"KB {entry['id']}: unknown conflict_id {conflict_id}"
                    )
            constraints = entry.get("constraints") or {}
            for key in ("must_have_any", "must_have_all", "must_not_have"):
                if key in constraints and not isinstance(constraints[key], list):
                    raise ValueError(
                        f"KB {entry['id']}: constraints.{key} must be a list"
                    )

    def _build_indexes(self) -> None:
        passages: list[str] = []
        docs: list[list[str]] = []
        negative_passages: list[str] = []
        self.family_vocabulary = {}
        self.phrase_records = []
        self.negative_phrase_records = []
        self.category_phrase_indices = {}
        self.category_negative_indices = {}
        for entry in self.entries:
            for phrase in entry["positive_examples"]:
                idx = len(passages)
                self.phrase_records.append({"entry_id": entry["id"], "phrase": phrase, "index": idx})
                self.category_phrase_indices.setdefault(entry["id"], []).append(idx)
                passages.append(f"passage: {phrase}")
                docs.append(tokenize(phrase))

            # Negative examples are not retrieval documents. They are a separate
            # semantic contradiction channel used after candidate generation.
            negatives = list(entry.get("negative_phrases", [])) + list(entry.get("hard_negative_examples", []))
            seen_neg: set[str] = set()
            for phrase in negatives:
                if not isinstance(phrase, str):
                    continue
                key = normalize_light(phrase)
                if not key or key in seen_neg:
                    continue
                seen_neg.add(key)
                idx = len(negative_passages)
                self.negative_phrase_records.append({"entry_id": entry["id"], "phrase": phrase, "index": idx})
                self.category_negative_indices.setdefault(entry["id"], []).append(idx)
                negative_passages.append(f"passage: {phrase}")

            family = str(entry.get("family") or "other")
            vocab = self.family_vocabulary.setdefault(family, set())
            for phrase in entry.get("positive_examples", [])[:40]:
                vocab.update(tokenize(phrase))
            for phrase in (entry.get("intent_signals") or {}).get("positive", [])[:30]:
                vocab.update(tokenize(str(phrase)))

            prototype = self._category_prototype(entry)
            emb = self.model.encode([f"passage: {prototype}"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            self.category_embeddings[entry["id"]] = np.asarray(emb, dtype=np.float32)

        self.phrase_embeddings_matrix = (
            np.asarray(self.model.encode(passages, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=64), dtype=np.float32)
            if passages else np.empty((0, 0), dtype=np.float32)
        )
        self.negative_embeddings_matrix = (
            np.asarray(self.model.encode(negative_passages, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=64), dtype=np.float32)
            if negative_passages else np.empty((0, 0), dtype=np.float32)
        )
        self.intent_docs = [tokenize(" ".join(entry["positive_examples"])) for entry in self.entries]
        self.bm25 = BM25Index(self.intent_docs)
        self.intent_index = {entry["id"]: i for i, entry in enumerate(self.entries)}

    @staticmethod
    def _category_prototype(entry: dict) -> str:
        category = str(entry.get("category") or "")
        family = str(entry.get("family") or "")
        definition = str(entry.get("definition") or entry.get("scope") or entry.get("description") or "")
        positives = ". ".join(str(x) for x in entry.get("positive_examples", [])[:10])
        intent_type = str(entry.get("intent_type") or "")
        facets = ", ".join(str(x) for x in entry.get("intent_facets", [])[:8])
        signals = entry.get("intent_signals") or {}
        signal_pos = ". ".join(str(x) for x in signals.get("positive", [])[:10])
        return (
            f"Семейство: {family}. Категория: {category}. "
            f"Тип интента: {intent_type}. Фасеты: {facets}. "
            f"Определение: {definition}. Положительные примеры: {positives}. "
            f"Intent-сигналы: {signal_pos}."
        )

    # --------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------

    def get_by_id(self, kb_id: str) -> dict | None:
        return self.entries_by_id.get(str(kb_id))


    def query_quality(self, question: str) -> QueryQuality:
        """Expose the bounded input-quality assessment for diagnostics."""
        return _query_quality(question)

    # --------------------------------------------------------
    # HARD RULES
    # --------------------------------------------------------

    def _hard_rule_match(self, question: str) -> str | None:
        """Return a deterministic intent only when the wording contains a strong
        business signal. Rules intentionally use token-prefix checks for Russian
        inflection/order variation instead of requiring an exact phrase.
        """
        q = normalize_light(question)
        tokens = q.split()

        def has_any(*prefixes: str) -> bool:
            return any(token.startswith(prefix) for token in tokens for prefix in prefixes)

        stems = stem_tokens(q)

        def has_phrase(*phrases: str) -> bool:
            return any(normalize_light(p) in q for p in phrases)

        def has_stem(*prefixes: str) -> bool:
            return any(s.startswith(prefixes) for s in stems)

        def has_school() -> bool:
            return any(token.startswith("школ") or light_stem(token).startswith("школ") for token in tokens)

        def has_flow() -> bool:
            return any(token.startswith(("поток", "потоков")) for token in tokens)

        # ----------------------------------------------------
        # School lifecycle: the most important conflict group.
        # ----------------------------------------------------
        explicit_transfer = (
            has_any("смен") and has_school()
        ) or (
            has_any("перейд", "перевод", "перевест") and has_school()
        )
        explicit_return = has_any("восстанов", "верн", "обратн") and (
            has_school() or has_any("отчис", "исключ", "ушел", "ушла", "вышел", "вышла")
        )
        explicit_exclusion = has_any("исключ", "выгнал", "отчисл") and (
            has_phrase("меня", "мне") or has_any("лиш")
        )

        # Exit from all schools must win over generic voluntary-exit signals.
        if has_phrase(
            "из всех школ", "из всех направлений", "из всего обучения",
            "отовсюду", "нигде не учиться",
        ):
            return "kb_024"

        # Voluntary exit must be checked before generic school/education retrieval.
        # `отчислиться` is voluntary; `меня отчислили` is an already completed exclusion.
        voluntary_exit = (
            has_any("выйт", "уйт", "покин", "прекрат", "удалит", "убер") and has_school()
        ) or has_phrase(
            "отчислиться из школы",
            "хочу отчислиться из школы",
            "как отчислиться из школы",
        )

        if explicit_transfer:
            return "kb_006"
        if explicit_return:
            return "kb_008"
        if explicit_exclusion and not voluntary_exit:
            return "kb_007"
        if voluntary_exit and not explicit_transfer and not explicit_return:
            return "kb_005"

        # ----------------------------------------------------
        # Mentor family.
        # The critical rule is: the mere presence of «наставник» must NEVER
        # select the personal-mentor category. First determine the user's
        # role/action: become a mentor, ask about the programme, retry an
        # application, or ask about an already assigned mentor.
        # ----------------------------------------------------
        mentor = has_any("настав", "ментор")
        become_stems = {"стат", "хот", "жел", "готов", "мог"}
        explicit_role_negation = mentor and (
            has_phrase(
                "не хочу быть наставником", "не хочу стать наставником",
                "не хочу быть ментором", "не хочу стать ментором",
                "не желаю быть наставником", "не желаю стать наставником",
                "не готов быть наставником", "не готов стать наставником",
            )
            or bool(re.search(
                r"\bне\s+(?:хоч|жел|готов)\w*\s+(?:быть|стать)\s+(?:наставник\w*|ментор\w*)",
                q,
            ))
        )
        wants_to_become = mentor and not explicit_role_negation and (
            has_any("хот", "жел", "готов", "стат", "мог") and
            has_stem("стат", "хоч", "хот", "жел", "готов", "мог")
            or has_phrase(
                "хочу быть наставником", "хочу стать наставником",
                "как стать наставником", "как быть наставником",
                "хотел стать наставником", "хотела стать наставником",
                "хотел бы стать наставником", "хотела бы стать наставником",
                "хочу быть ментором", "хочу стать ментором",
            )
        )
        # Common inflected / future-role wording, e.g.
        # «хочу быть наставников в будущем» / «хочу стать наставником потом».
        future_role = mentor and has_any("будущ", "потом", "дальш") and has_any("хот", "жел", "готов", "стат", "буд")
        wants_to_become = wants_to_become or future_role

        personal_mentor_problem = mentor and (
            has_phrase(
                "мой наставник", "моего наставника", "моему наставнику",
                "мой ментор", "моего ментора", "кто мой наставник",
                "как найти наставника", "как связаться с наставником",
                "наставник не отвечает", "нет наставника",
                "не назначили наставника", "сменить наставника",
            )
        )
        retry_signal = mentor and has_any("повтор", "снов", "отказ", "отклон")
        program_signal = mentor and has_any("программ", "наставничеств")

        # This is intentionally before the quality gate in match():
        # short requests such as «Хочу стать наставником» are valid.
        if wants_to_become and not personal_mentor_problem and not program_signal and not retry_signal:
            return "kb_023"
        if program_signal and not wants_to_become and not personal_mentor_problem and not retry_signal:
            return "kb_034"
        if retry_signal and not wants_to_become and not personal_mentor_problem:
            return "kb_025"
        if personal_mentor_problem and not wants_to_become and not retry_signal:
            return "kb_009"
        # Ambiguous mentor mention: do NOT guess. Let retrieval/choice/LLM
        # decide from a broader evidence set.

        # ----------------------------------------------------
        # Education entry / school registration / registration failure.
        # ----------------------------------------------------
        registration_terms = has_any("запис", "вступ", "попаст", "оформ", "регистра")
        failure_terms = has_any("не", "нет", "ошиб", "проблем", "неприн", "отказ")
        registration_failure = registration_terms and (
            failure_terms
            or has_phrase("не пускает в школу", "нет кнопки записи", "кнопки записи нет", "не дает записаться")
        )
        school_registration = has_school() and has_any("запис", "вступ", "попаст", "выбрат", "оформ")
        general_training = has_any("обуч", "учит", "учеб", "курс")

        # "Не получается записаться" is a complete supported intent even
        # without the word "школа"; don't require a noun the user may omit.
        if registration_failure:
            return "kb_002"
        if school_registration:
            return "kb_033"

        # Explicit general training entry.
        if general_training and has_any("попаст", "начат", "старт", "попад", "участв", "хот", "нуж"):
            return "kb_035"
        if general_training and registration_terms and not has_school():
            return "kb_035"

        # "Хочу попасть/начать" without an explicit school is the broader
        # education-entry intent, not school registration.
        if has_any("обуч", "учит", "учеб") and has_any("попаст", "начат", "старт", "попад"):
            return "kb_035"

        # ----------------------------------------------------
        # Current / next stream boundaries.
        # ----------------------------------------------------
        already_started = (
            (has_any("уже", "опоздал", "текущ") and has_any("начал", "идет", "старт", "поток", "школ"))
            or has_phrase("школа уже началась", "уже идет поток", "уже стартовала школа")
        )
        if already_started:
            return "kb_015"

        asks_flow_date = (
            has_flow() and
            (has_any("когда", "дат", "старт", "запуск", "начн", "следующ", "нов") or any(ch.isdigit() for ch in q))
        )
        if asks_flow_date:
            return "kb_016"

        # ----------------------------------------------------
        # App/account failure. Prefer this over broad semantic retrieval when
        # the user explicitly says the application rejects their data or login.
        # ----------------------------------------------------
        app_signal = has_any("прилож", "приложен")
        login_signal = has_any("зайт", "войд", "вход", "авториз", "логин")
        rejected_data = has_any("данн", "информац", "учет") and failure_terms
        explicit_app_login = (
            app_signal
            and (
                has_phrase("не могу войти", "не могу зайти", "не удается войти",
                           "не удаётся войти", "не могу авторизоваться")
                or login_signal
            )
        )
        if (explicit_app_login and failure_terms) or rejected_data:
            return "kb_001"

        # ----------------------------------------------------
        # Account / phone verification (before generic "код").
        # Narrow: SMS / confirmation code for phone number only.
        # ----------------------------------------------------
        phone_signal = has_any("телефон", "номер") or has_phrase(
            "подтверждения номера", "подтвердить номер", "подтверждение номера",
        )
        sms_signal = has_any("смс", "sms") or has_phrase(
            "не приходит смс", "смс с кодом", "смс код",
        )
        code_confirm = has_phrase(
            "код на телефон",
            "код подтверждения",
            "не приходит код",
            "код подтверждения номера",
        )
        facility_code_noise = has_any("калит", "домофон", "двер") or has_phrase(
            "код калитки", "код двери", "код домофона",
        )
        invite_or_qr_noise = has_any("инвайт", "qr") or has_phrase("код приглашения")
        # SMS + code in this product almost always means phone verification.
        phone_verification = (
            (phone_signal and (sms_signal or code_confirm or has_any("код")))
            or (sms_signal and has_any("код"))
            or (phone_signal and code_confirm)
        )
        if phone_verification and not facility_code_noise and not invite_or_qr_noise:
            return "kb_027"

        # Facility / gate / door code (physical entrance).
        if (
            has_any("калит", "домофон")
            or (
                has_any("код")
                and has_any("двер", "вход")
                and not has_any("инвайт", "qr", "приглаш", "смс", "sms", "телефон")
            )
            or has_phrase("код калитки", "код двери", "код домофона", "пароль от двери")
        ):
            if not has_any("инвайт", "qr", "смс", "sms") and not has_phrase("код приглашения"):
                return "kb_019"

        # No access to schools section in the app.
        if (
            (has_any("нет", "не") and has_any("доступ") and has_school())
            or has_phrase(
                "не вижу школы",
                "не открываются школы",
                "нет школ в приложении",
                "нет доступа к школам",
                "не вижу школы в приложении",
                "закрыты школы",
            )
        ) and not has_any("инвайт", "qr", "запис", "смс", "телефон"):
            return "kb_017"

        # ----------------------------------------------------
        # Telegram / codes / QR.
        # ----------------------------------------------------
        if (has_any("инвайт") or has_phrase("код приглашения")) and not has_any("qr"):
            return "kb_013"
        if has_any("qr"):
            return "kb_014"
        if has_any("не добав", "добавь", "не включ") and has_any("чат", "групп", "телеграм", "тг"):
            return "kb_021"
        if has_any("ссылк") and has_any("чат", "телеграм", "тг", "групп"):
            return "kb_004"

        # Fall back to the explicitly curated rules only when exactly one rule fires.
        matches: list[tuple[str, int, int]] = []
        for entry_id, positives, negatives in HARD_INTENT_RULES:
            # A negated role-change statement such as «не хочу быть наставником»
            # must never fall through to the generic positive phrase table.
            if entry_id == "kb_023" and explicit_role_negation:
                continue
            if not any(normalize_light(p) in q for p in positives):
                continue
            if any(normalize_light(n) in q for n in negatives):
                continue
            best_trigger_len = max(
                (len(normalize_light(p)) for p in positives if normalize_light(p) in q),
                default=0,
            )
            priority = int(self.entries_by_id.get(entry_id, {}).get("priority", 0))
            matches.append((entry_id, best_trigger_len, priority))

        if matches:
            matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
            if len(matches) == 1 or matches[0][1] > matches[1][1] + 4:
                return matches[0][0]

        return None

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def _semantic_scores(self, question: str) -> dict[str, dict[str, float]]:
        if self.phrase_embeddings_matrix.size == 0:
            return {}
        query_emb = self.model.encode(
            f"query: {question}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        phrase_scores = np.dot(self.phrase_embeddings_matrix, query_emb)
        negative_scores = (
            np.dot(self.negative_embeddings_matrix, query_emb)
            if self.negative_embeddings_matrix.size else np.empty((0,), dtype=np.float32)
        )
        result: dict[str, dict[str, float]] = {}
        for entry in self.entries:
            indices = self.category_phrase_indices.get(entry["id"], [])
            vals = np.asarray([phrase_scores[i] for i in indices], dtype=np.float32)
            if vals.size == 0:
                continue
            vals_sorted = np.sort(vals)[::-1]
            top3 = float(np.mean(vals_sorted[: min(3, len(vals_sorted))]))
            max_score = float(vals_sorted[0])
            proto = float(np.dot(self.category_embeddings[entry["id"]], query_emb))
            neg_indices = self.category_negative_indices.get(entry["id"], [])
            neg_max = float(max((negative_scores[i] for i in neg_indices), default=0.0))
            result[entry["id"]] = {
                "max": max_score,
                "top3": top3,
                "prototype": proto,
                "negative_max": neg_max,
            }
        return result

    def _lexical_scores(self, question: str) -> tuple[np.ndarray, np.ndarray]:
        q_tokens = tokenize(question)
        raw = self.bm25.score(q_tokens)
        # Robust normalization: rank/relative scale instead of raw threshold.
        if raw.size:
            max_raw = float(raw.max())
            normalized = raw / max_raw if max_raw > 0 else raw
        else:
            normalized = raw
        return raw, normalized

    def _retrieve_candidates(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        semantic = self._semantic_scores(question)
        lexical_raw, lexical_norm = self._lexical_scores(question)

        semantic_rank = sorted(
            range(len(self.entries)),
            key=lambda i: (
                0.5 * semantic.get(self.entries[i]["id"], {}).get("max", 0.0)
                + 0.3 * semantic.get(self.entries[i]["id"], {}).get("top3", 0.0)
                + 0.2 * semantic.get(self.entries[i]["id"], {}).get("prototype", 0.0)
            ),
            reverse=True,
        )
        lexical_rank = sorted(range(len(self.entries)), key=lambda i: float(lexical_norm[i]), reverse=True)
        fused = reciprocal_rank_fusion([semantic_rank[:top_k], lexical_rank[:top_k]])

        candidate_indices = sorted(fused, key=fused.get, reverse=True)[:top_k]
        result: list[dict] = []
        for i in candidate_indices:
            entry = self.entries[i]
            sem = semantic.get(entry["id"], {"max": 0.0, "top3": 0.0, "prototype": 0.0, "negative_max": 0.0})
            phrase_idx = self._best_phrase_index(entry["id"], question)
            result.append({
                "entry": entry,
                "embedding_score": float(sem["max"]),
                "semantic_top3": float(sem["top3"]),
                "semantic_prototype": float(sem["prototype"]),
                "negative_semantic_score": float(sem.get("negative_max", 0.0)),
                "bm25_raw": float(lexical_raw[i]),
                "bm25_score": float(lexical_norm[i]),
                "rrf_score": float(fused.get(i, 0.0)),
                "best_phrase": phrase_idx["phrase"] if phrase_idx else "",
            })
        return result

    def _best_phrase_index(self, entry_id: str, question: str) -> dict | None:
        q = normalize_light(question)
        best = None
        best_score = -1.0
        entry = self.entries_by_id[entry_id]
        for phrase in entry.get("positive_examples", []):
            score = fuzz.token_set_ratio(q, normalize_light(phrase))
            if score > best_score:
                best_score = score
                best = {"phrase": phrase, "score": score}
        return best

    # --------------------------------------------------------
    # EVIDENCE / RANKING
    # --------------------------------------------------------

    def _lexical_phrase_hits(self, question: str, entry: dict) -> list[str]:
        q = normalize_light(question)
        hits = []
        for phrase in entry.get("positive_examples", []):
            p = normalize_light(phrase)
            if p and p in q:
                hits.append(phrase)
        return hits

    def _best_fuzzy_phrase_score(self, question: str, entry: dict) -> float:
        """Best phrase-level lexical similarity on a 0..1 scale.

        This is intentionally a secondary signal. It helps with word-order
        variation and minor typos, but cannot override semantic contradictions.
        """
        q = normalize_light(question)
        best = 0.0
        for phrase in entry.get("positive_examples", []):
            p = normalize_light(phrase)
            if not p:
                continue
            score = max(
                fuzz.token_set_ratio(q, p),
                fuzz.partial_ratio(q, p),
            ) / 100.0
            best = max(best, float(score))
        return best

    def _domain_support(self, question: str, entry: dict) -> float:
        """How much concrete, category-relevant support exists in the query.

        This is intentionally different from semantic similarity. It answers
        whether the query contains any explicit business/domain evidence that
        can be aligned with the candidate. A high embedding score alone is not
        domain proof.
        """
        qo = self._query_ontology(question)
        ontology = entry.get("ontology") or self._derive_ontology(entry)
        q_tokens = set(tokenize(question))
        family = str(entry.get("family") or "other")
        vocab = self.family_vocabulary.get(family, set())

        signals = 0.0
        if qo["object"] and ontology.get("object") in qo["object"]:
            signals += 0.38
        if qo["action"] and ontology.get("action") in qo["action"]:
            signals += 0.22
        if qo["state"] and ontology.get("state") in qo["state"]:
            signals += 0.18
        if qo["facet"] and ontology.get("facet") in qo["facet"]:
            signals += 0.16

        if vocab and q_tokens:
            direct = len(q_tokens & vocab)
            if direct:
                signals += min(0.22, 0.07 * direct)

        if self._lexical_phrase_hits(question, entry):
            signals += 0.30
        if self._required_evidence(question, entry).get("coverage", 0.0) >= 0.50:
            signals += 0.18

        return float(min(1.0, signals))

    def _ontology_alignment(self, question: str, entry: dict) -> tuple[float, str]:
        """Compare explicit query meaning with candidate ontology.

        Explicit mismatches are hard evidence against a candidate. Unknown
        dimensions remain neutral, so recall is preserved for terse queries.
        """
        qo = self._query_ontology(question)
        ontology = entry.get("ontology") or self._derive_ontology(entry)
        checks: list[tuple[str, float, str]] = []

        dimensions = ("object", "action", "state", "facet")
        for dim in dimensions:
            q_values = qo.get(dim) or set()
            c_value = ontology.get(dim)
            if not q_values or not c_value:
                continue
            if c_value in q_values:
                checks.append((dim, 1.0, "match"))
            else:
                checks.append((dim, 0.0, f"mismatch:{','.join(sorted(q_values))}!=${c_value}"))

        if not checks:
            return 0.50, "no_explicit_ontology_signal"

        weights = {"object": 0.42, "action": 0.20, "state": 0.18, "facet": 0.20}
        numerator = sum(weights[d] * v for d, v, _ in checks)
        denominator = sum(weights[d] for d, _, _ in checks)
        score = numerator / max(denominator, 1e-9)
        mismatches = [d for d, v, _ in checks if v == 0.0]
        if mismatches:
            return float(score), "ontology_mismatch:" + ",".join(mismatches)
        return float(score), "ontology_aligned:" + ",".join(d for d, _, _ in checks)

    def _external_target_hint(self, question: str) -> str | None:
        """Detect only explicit third-party targets in domain-sensitive phrases."""
        raw = (question or "").strip()
        q = normalize_light(raw)
        if not any(ctx in q for ctx in EXTERNAL_TARGET_CONTEXT):
            return None
        for match in EXTERNAL_TARGET_RE.finditer(raw):
            entity = normalize_light(match.group("entity"))
            if entity and entity not in KNOWN_INTERNAL_ENTITIES:
                return entity
        return None

    def _scope_evidence(self, question: str, ranked: list[dict]) -> dict[str, Any]:
        """Assess whether the message has defensible evidence of belonging to our support domain.

        Scope is deliberately separate from intent ranking. A semantically similar but
        unsupported message must not be forced into the nearest KB category.
        """
        qo = self._query_ontology(question)
        normalized = normalize_light(question)
        tokens = set(tokenize(question))

        domain_hits = 0
        for token in tokens:
            if any(token.startswith(prefix) for prefix in DOMAIN_TERMS):
                domain_hits += 1

        direct_supported = False
        strong_intent_evidence = 0.0
        best_domain_support = 0.0
        best_family_fit = 0.0
        best_score = 0.0
        best_semantic = 0.0
        best_ontology = 0.0

        for item in ranked[:8]:
            best_domain_support = max(best_domain_support, float(item.get("domain_support_score", 0.0)))
            best_family_fit = max(best_family_fit, float(item.get("family_fit_score", 0.0)))
            best_score = max(best_score, float(item.get("final_score", 0.0)))
            best_semantic = max(best_semantic, float(item.get("semantic_score", 0.0)))
            best_ontology = max(best_ontology, float(item.get("ontology_alignment_score", 0.0)))
            if item.get("direct_support") or item.get("positive_hits") or item.get("intent_signal_positive_hits"):
                direct_supported = True
            strong_intent_evidence = max(strong_intent_evidence, float(item.get("real_intent_evidence", 0.0)))

        external_target = self._external_target_hint(question)
        known_ontology = any(qo.get(dim) for dim in ("object", "action", "state", "facet"))

        # Explicit internal domain vocabulary is meaningful even when a category is
        # not identifiable yet. This protects rare/novel but legitimate support queries.
        in_domain_signal = bool(domain_hits or known_ontology or external_target is None and best_family_fit >= 0.30)

        return {
            "domain_hits": domain_hits,
            "known_ontology": known_ontology,
            "external_target": external_target,
            "best_domain_support": best_domain_support,
            "best_family_fit": best_family_fit,
            "best_score": best_score,
            "best_semantic": best_semantic,
            "best_ontology": best_ontology,
            "strong_intent_evidence": strong_intent_evidence,
            "direct_supported": direct_supported,
            "in_domain_signal": in_domain_signal,
        }

    def _oos_reason(self, question: str, ranked: list[dict], quality: QueryQuality) -> str | None:
        """Return an OOS reason only for coherent messages with weak supported-domain evidence.

        Priority order:
        1) explicit third-party target in an otherwise domain-sensitive phrase;
        2) no domain/ontology evidence and no candidate intent evidence;
        3) low-score nearest neighbour with no direct support.
        """
        if not ranked or quality.meta_noise:
            return None

        scope = self._scope_evidence(question, ranked)
        external_target = scope["external_target"]

        if (
            OOS_EXTERNAL_TARGET
            and external_target
            and not scope["direct_supported"]
            and scope["best_domain_support"] < 0.40
            and scope["strong_intent_evidence"] < 0.42
        ):
            return "oos:external_target"

        # Never call a message OOS when there is any concrete KB evidence.
        if (
            scope["direct_supported"]
            or scope["strong_intent_evidence"] >= 0.34
            or scope["known_ontology"]
            or scope["domain_hits"] > OOS_MIN_DOMAIN_TERM_HITS
            or scope["best_domain_support"] >= OOS_DOMAIN_SUPPORT_MAX
            or scope["best_family_fit"] >= 0.42
        ):
            return None

        coherent_for_oos = (
            quality.score >= OOS_QUERY_QUALITY_MIN
            or (quality.tokens >= 3 and quality.question_signal >= 0.35 and not quality.meta_noise)
        )
        if (
            coherent_for_oos
            and scope["best_score"] <= OOS_TOP_SCORE_MAX
            and scope["strong_intent_evidence"] <= OOS_REAL_EVIDENCE_MAX
            and scope["best_domain_support"] <= OOS_DOMAIN_SUPPORT_MAX
            and scope["best_semantic"] < 0.82
            and scope["best_ontology"] < 0.62
        ):
            return "oos:no_supported_intent"

        return None

    def _negative_decision_reason(self, question: str, top: dict, ranked: list[dict]) -> str | None:
        """Return a safe abstention reason when the candidate is contradicted.

        This runs after retrieval/ranking. It does not replace positive intent
        classification; it prevents semantically-close but structurally-wrong
        categories from being auto-accepted.
        """
        entry = top["entry"]
        qo = top.get("query_ontology") or self._query_ontology(question)
        ontology = top.get("ontology") or self._derive_ontology(entry)
        alignment = float(top.get("ontology_alignment_score", 0.50))
        domain_support = float(top.get("domain_support_score", 0.0))
        contrast = float(top.get("negative_contrast_score", 0.0))
        direct = bool(top.get("direct_support"))

        if top.get("negative_hits") or (top.get("constraint_evidence") or {}).get("forbidden_hits"):
            return "explicit_candidate_negative"

        if qo.get("object") and alignment <= ONTOLOGY_HARD_MISMATCH:
            return "unsupported_object" if not any(
                (item.get("ontology") or {}).get("object") in qo["object"]
                for item in ranked
            ) else "object_mismatch"

        if qo.get("facet") and alignment <= ONTOLOGY_HARD_MISMATCH:
            supported_facets = set()
            for item in ranked:
                supported_facets.add((item.get("ontology") or {}).get("facet"))
            if not any(f in supported_facets for f in qo["facet"]):
                return "unsupported_facet"
            return "facet_mismatch"

        if self._external_target_hint(question):
            # A third-party target is only a hard negative when the query is
            # otherwise asking for a domain operation where the current KB is
            # clearly tied to the internal entity.
            if ontology.get("object") in {"training", "school", "mentor_program"}:
                return "wrong_entity"

        if contrast <= NEGATIVE_CONTRAST_UNSAFE:
            return "negative_evidence_dominates"

        if not direct and domain_support < MIN_DOMAIN_SUPPORT_FOR_AUTO and alignment < ONTOLOGY_ALIGNMENT_MIN:
            return "insufficient_domain_support"

        return None

    def _family_fit(self, question: str, entry: dict) -> float:
        """Family fit combines lexical family evidence with ontology compatibility."""
        family = str(entry.get("family") or "other")
        vocab = self.family_vocabulary.get(family, set())
        q_tokens = set(tokenize(question))
        lexical = 0.0
        if vocab and q_tokens:
            direct = len(q_tokens & vocab)
            q_stems = {light_stem(token) for token in q_tokens}
            vocab_stems = {light_stem(token) for token in vocab}
            stemmed = len(q_stems & vocab_stems)
            lexical = min(1.0, (direct + 0.70 * max(0, stemmed - direct)) / max(1.0, min(4.0, len(q_tokens))))
        qo = self._query_ontology(question)
        ontology = entry.get("ontology") or self._derive_ontology(entry)
        obj = ontology.get("object")
        object_hints = qo["object"]
        # Strong explicit object evidence should dominate weak family token overlap.
        object_fit = 0.5
        if object_hints:
            if obj in object_hints:
                object_fit = 1.0
            elif obj == "training" and "training" in object_hints:
                object_fit = 1.0
            elif obj not in object_hints:
                object_fit = 0.0
        return float(min(1.0, 0.55 * lexical + 0.45 * object_fit))

    # --------------------------------------------------------
    # INTENT FIT / QUESTION-TYPE COMPATIBILITY
    # --------------------------------------------------------

    @staticmethod
    def _question_facets(question: str) -> set[str]:
        """Detect strong question-type signals without requiring ML training.

        This is deliberately conservative: a facet is emitted only for strong,
        phrase-level indicators. These signals are used as a safety/fit layer,
        not as a replacement for semantic retrieval.
        """
        q = normalize_light(question)
        facets: set[str] = set()

        # Morphology-aware role transition detection. Exact phrase lists are
        # not enough for Russian inflections such as «наставников», «ментором»
        # and natural variants like «хочу быть … в будущем».
        stems = set(stem_tokens(question))
        mentor_stems = {"наставник", "ментор"}
        wants_stems = {"хоч", "хочу", "жел", "готов", "мог"}
        become_stems = {"стат"}
        if stems & mentor_stems and ((stems & wants_stems) and (stems & become_stems)):
            facets.add("mentor_application")
        if stems & mentor_stems and "будущ" in stems and (stems & wants_stems):
            facets.add("mentor_application")

        patterns: dict[str, tuple[str, ...]] = {
            "mentor_application": (
                "хочу стать наставником", "хотел бы стать наставником",
                "хотела бы стать наставником", "хочу быть наставником",
                "стать ментором", "хочу быть ментором",
                "как стать наставником", "как быть наставником",
                "хочу преподавать",
            ),
            "mentor_program": (
                "программа наставничества", "как работает наставничество",
                "что такое наставничество",
            ),
            "personal_mentor": (
                "мой наставник", "моего наставника", "кто мой наставник",
                "мой ментор", "наставник не отвечает",
            ),
            "retry": (
                "повторно подать", "заново подать", "повторная заявка",
                "заявку отклонили", "заявка отклонена",
            ),
            "school": (
                "выбрать школу", "в школу", "школу выбрать",
            ),
            "registration": (
                "записаться", "запись", "регистрация", "зарегистрироваться",
            ),
            "voluntary_exit": (
                "выйти из школы", "уйти из школы", "отчислиться из школы",
                "покинуть школу",
            ),
            "school_transfer": (
                "сменить школу", "перейти в другую школу",
                "перевестись в другую школу",
            ),
            "restoration": (
                "восстановиться в школе", "вернуться в школу",
                "восстановить обучение",
            ),
            "involuntary_exclusion": (
                "меня исключили", "меня отчислили", "меня выгнали",
            ),
            "telegram_membership": (
                "добавьте меня в чат", "не добавили в чат",
                "нет меня в группе", "не добавили в телеграм",
            ),
            "telegram_link": (
                "ссылка на чат", "ссылка на телеграм",
                "ссылка не открывается",
            ),
            "invite_code": (
                "инвайт код", "инвайт-код", "код приглашения",
            ),
            "qr_code": (
                "qr код", "qr-код", "не сканируется qr",
            ),
            "account": (
                "не принимает данные", "регистрация с разными аккаунтами",
                "не тот аккаунт",
            ),
            "verification": (
                "код подтверждения", "не приходит смс", "подтвердить номер",
            ),
            "age": (
                "с какого возраста", "до какого возраста", "со скольки лет",
                "с какого лет", "после 50", "после 40", "после 30",
                "сколько лет можно", "возраст", "лет можно учиться",
            ),
            "duration": (
                "сколько длится", "как долго длится", "как долго идет",
                "как долго обучение", "сколько месяцев", "сколько недель",
                "сколько по времени", "длительность обучения", "продолжительность обучения",
            ),
            "price": (
                "сколько стоит", "стоимость", "цена", "платно ли", "платное ли",
                "нужно ли платить", "надо ли платить", "бесплатно ли",
            ),
            "date": (
                "когда следующий", "когда новый", "когда старт", "когда запуск",
                "какая дата", "дата старта", "даты потока", "в какой день",
            ),
            "procedure": (
                "как попасть", "как записаться", "как начать", "что нужно чтобы",
                "как вступить", "как стать участником", "как учиться",
            ),
            "location": (
                "где находится", "где найти", "где посмотреть", "куда прийти",
                "где выбрать", "где взять", "где получить",
            ),
            "technical_failure": (
                "не работает", "не открывается", "не запускается", "ошибка",
                "не загружается", "не сканируется", "не принимает", "не могу войти",
            ),
            "access": (
                "нет доступа", "не пускает", "не вижу", "закрыт доступ", "как получить доступ",
            ),
        }
        for facet, phrases in patterns.items():
            if any(p in q for p in phrases):
                facets.add(facet)

        # The existing phrase detector intentionally remains conservative; this
        # second pass covers the most important Russian inflection variants.
        if stems & {"наставник", "ментор"} and (stems & {"хоч", "хочу", "жел", "готов", "мог"}):
            if "стат" in stems or ("будущ" in stems and "быть" in stems):
                facets.add("mentor_application")

        if stems & {"наставник", "ментор"} and stems & {"отказ", "отклон", "повтор", "снов"}:
            facets.add("retry")

        # Morphological variants which are common in real support messages but
        # are not covered by exact phrase lists. These remain intent-specific
        # and do not classify from a generic topic word alone.
        if any(stem.startswith("запис") or stem.startswith("регистрац") for stem in stems):
            facets.add("registration")
        if any(stem.startswith("ссыл") for stem in stems) and (stems & {"чат", "телеграм", "тг"}):
            facets.add("telegram_link")
        if stems & {"наставник", "ментор"} and "mentor_application" not in facets and "mentor_program" not in facets:
            mentor_text = normalize_light(question)
            if not any(p in mentor_text for p in (
                "стать наставником", "стать ментором", "программа наставничества",
                "как работает наставничество", "что такое наставничество",
            )):
                facets.add("personal_mentor")

        return facets

    @staticmethod
    def _entry_supported_facets(entry: dict) -> set[str]:
        """Infer which strong question types an intent's examples support."""
        texts: list[str] = []
        texts.extend(str(x) for x in entry.get("positive_examples", [])[:30])
        texts.extend(str(x) for x in entry.get("when_to_use", [])[:10])
        return KnowledgeBase._question_facets(". ".join(texts))

    def _intent_fit(self, question: str, entry: dict, *, fuzzy_phrase: float = 0.0) -> tuple[float, str]:
        """Score how well the *intent contract* of an entry matches the query.

        Semantic similarity answers "what is this text about?".
        Intent fit answers "does this question ask for the kind of action/state
        this category is designed to handle?".

        Explicit KB intent signals take precedence over inferred facets.
        """
        q = normalize_light(question)
        q_facets = self._question_facets(question)

        intent_type = str(entry.get("intent_type") or "").strip()
        required_facets = INTENT_REQUIRED_FACETS.get(intent_type, set())
        # For critical near-neighbour intents, topic overlap is not enough.
        # Missing the defining facet means the category is unsupported even if
        # embeddings/lexical retrieval rank it highly.
        if required_facets and not (q_facets & required_facets):
            return 0.08, (
                f"missing_required_intent_facet:{intent_type};"
                f"required={','.join(sorted(required_facets))}"
            )

        explicit_facets = {
            str(x).strip()
            for x in entry.get("intent_facets", [])
            if isinstance(x, str) and str(x).strip()
        }
        supported = explicit_facets or self._entry_supported_facets(entry)

        signals = entry.get("intent_signals") or {}
        positive_signals = [
            normalize_light(str(x))
            for x in signals.get("positive", [])
            if isinstance(x, str) and normalize_light(str(x))
        ]
        negative_signals = [
            normalize_light(str(x))
            for x in signals.get("negative", [])
            if isinstance(x, str) and normalize_light(str(x))
        ]

        positive_hits = [x for x in positive_signals if x in q]
        negative_hits = [x for x in negative_signals if x in q]

        # A direct positive intent signal is strong evidence, but not a hard gate.
        if positive_hits and not negative_hits:
            signal_strength = min(1.0, 0.55 + 0.12 * len(positive_hits))
        elif positive_hits and negative_hits:
            signal_strength = 0.50
        else:
            signal_strength = 0.0

        # Explicit negative intent signals are stronger than generic semantics.
        if negative_hits:
            fit = 0.12 + 0.10 * min(1.0, fuzzy_phrase)
            return float(min(fit, 0.25)), (
                "explicit_negative_signal:" + ",".join(negative_hits[:3])
            )

        if q_facets:
            overlap = q_facets & supported
            if overlap:
                facet_fit = 0.80 + 0.15 * (len(overlap) / max(1, len(q_facets)))
            else:
                facet_fit = 0.18
        else:
            facet_fit = 0.55 + 0.25 * min(1.0, fuzzy_phrase)

        # Blend explicit intent signals with facet compatibility.
        if signal_strength:
            fit = 0.60 * signal_strength + 0.40 * facet_fit
        else:
            fit = facet_fit

        # Structured intent compatibility catches semantic lookalikes such as
        # "стоимость вступления в клуб" vs "стоимость обучения".
        qo = self._query_ontology(question)
        ontology = entry.get("ontology") or self._derive_ontology(entry)
        object_hints = qo["object"]
        candidate_object = ontology.get("object")
        if object_hints and candidate_object:
            # Membership/club is a hard object distinction from training.
            if "membership" in object_hints and candidate_object != "membership":
                fit = min(fit, 0.28)
            elif candidate_object == "training" and object_hints & {"membership", "subscription", "rent"}:
                fit = min(fit, 0.28)
            elif candidate_object == "school" and "membership" in object_hints:
                fit = min(fit, 0.28)
        if "technical_failure" in q_facets and ontology.get("state") == "information":
            fit = min(fit, 0.30)
        if "price" in q_facets and entry.get("intent_type") in {"flow_date", "school_started"}:
            fit = min(fit, 0.25)

        # If the KB explicitly declares a single intent type and the query has
        # a strong, incompatible facet, keep the fit conservative.
        intent_type = str(entry.get("intent_type") or "").strip()
        strong_mismatch_pairs = {
            "education_entry": {"age", "duration", "price", "date"},
            "school_registration": {"age", "duration", "price", "date"},
            "education_price": {"age", "duration", "date"},
            "flow_date": {"age", "duration", "price"},
            "subscription_charge": {"age", "duration", "date"},
            "rent_payment_issue": {"age", "duration", "date", "price"},
        }
        if intent_type in strong_mismatch_pairs and q_facets & strong_mismatch_pairs[intent_type]:
            fit = min(fit, 0.22)
            return float(fit), (
                f"intent_type_mismatch:{intent_type};"
                f"query={','.join(sorted(q_facets & strong_mismatch_pairs[intent_type]))}"
            )

        fit = max(0.0, min(0.98, fit))
        if q_facets and not (q_facets & supported):
            reason = (
                f"facet_mismatch:query={','.join(sorted(q_facets))};"
                f"intent={','.join(sorted(supported)) or 'none'}"
            )
        elif positive_hits:
            reason = "positive_intent_signal:" + ",".join(positive_hits[:3])
        elif q_facets:
            reason = "facet_overlap:" + ",".join(sorted(q_facets & supported))
        else:
            reason = "no_strong_question_facet"
        return float(fit), reason

    def _structured_negation_hits(self, question: str, entry: dict) -> list[str]:
        """Detect explicit negation of the candidate's intended user goal.

        This is intentionally separate from `negative_phrases`: those are KB
        examples, while this detector reasons about *scope* around a matched
        action phrase. It must not turn failure reports ("не могу записаться")
        into a contradiction for the registration-failure intent.
        """
        q = normalize_light(question)
        if not q:
            return []

        positive_phrases: list[str] = []
        signals = entry.get("intent_signals") or {}
        positive_phrases.extend(
            normalize_light(str(x)) for x in signals.get("positive", [])
            if isinstance(x, str) and normalize_light(str(x))
        )
        positive_phrases.extend(
            normalize_light(str(x)) for x in entry.get("positive_examples", [])[:25]
            if isinstance(x, str) and normalize_light(str(x))
        )
        positive_phrases = list(dict.fromkeys(positive_phrases))

        hits: list[str] = []
        for phrase in positive_phrases:
            start = q.find(phrase)
            if start < 0:
                continue
            prefix = q[max(0, start - 40):start].strip()
            # A problem-state phrase is evidence for failure intents, not a
            # negation of the action itself.
            if any(prefix.endswith(p) for p in PROBLEM_STATE_NEGATION_PREFIXES):
                continue
            if any(prefix.endswith(p) for p in STRUCTURED_NEGATION_PREFIXES):
                hits.append(phrase)
        return list(dict.fromkeys(hits))

    def _semantic_precondition_reason(self, question: str, entry: dict) -> str | None:
        """Return a conservative precondition failure for auto-answering.

        Critical intents already declare `intent_type`/`intent_facets`. The new
        guard makes those contracts explicit at the final eligibility boundary,
        so retrieval similarity cannot bypass them.
        """
        intent_type = str(entry.get("intent_type") or "").strip()
        required_facets = INTENT_REQUIRED_FACETS.get(intent_type, set())
        q_facets = self._question_facets(question)
        if required_facets and not (q_facets & required_facets):
            return f"missing_required_intent_facet:{intent_type}"

        constraints = self._constraint_evidence(question, entry)
        any_groups = constraints.get("constraint_groups") or []
        if entry.get("constraints") and any_groups and all(isinstance(g, list) for g in any_groups):
            # The KB's `must_have_any` groups are intentionally treated as a
            # soft contract: at least one explicit constraint must be present.
            # This avoids rejecting natural variants while still blocking pure
            # topic overlap with no operational evidence.
            if not any(any(group) for group in any_groups):
                return "missing_semantic_precondition"

        if constraints.get("forbidden_hits"):
            return "forbidden_precondition"
        return None

    def _challenge_hard_rule(self, question: str, entry: dict) -> tuple[bool, str]:
        """Challenge a deterministic hard rule against close intent rivals.

        Hard rules are high precision hypotheses, not unconditional truth. A
        rival with a strong positive signal/facet fit can veto the shortcut and
        send the query through the normal contrastive ranker.
        """
        candidate_fit, _ = self._intent_fit(question, entry)
        candidate_family = self._family_fit(question, entry)
        candidate_pos, candidate_neg = self._intent_signal_hits(question, entry)
        candidate_score = (0.55 * candidate_fit) + (0.25 * candidate_family) + (0.20 if candidate_pos else 0.0)
        if candidate_neg:
            return False, "hard_rule_candidate_negative"

        conflict_ids = {str(x) for x in (entry.get("conflict_ids") or [])}
        rivals = [e for e in self.entries if str(e.get("id")) in conflict_ids]
        for rival in rivals:
            rival_fit, _ = self._intent_fit(question, rival)
            rival_family = self._family_fit(question, rival)
            rival_pos, rival_neg = self._intent_signal_hits(question, rival)
            if rival_neg:
                continue
            rival_score = (0.55 * rival_fit) + (0.25 * rival_family) + (0.20 if rival_pos else 0.0)
            if rival_score >= 0.66 and rival_score + 0.04 >= candidate_score:
                return False, f"hard_rule_challenged_by:{rival.get('id')}"
        return True, "hard_rule_challenge_passed"

    def _query_understanding(self, question: str) -> QueryUnderstanding:
        """Parse query meaning independently from candidate ranking.

        The parser is intentionally narrower than embeddings: it recognizes
        discriminating objects/facets, but leaves unknown objects unknown.
        """
        normalized = normalize_light(question)
        return understand_query(
            normalized,
            facets=self._question_facets(question),
            ontology=self._query_ontology(question),
        )

    def _intent_contract_evidence(self, question: str, entry: dict) -> ContractEvidence:
        """Verify whether the query proves the defining evidence of an intent.

        Retrieval is allowed to propose a candidate. This method decides whether
        there is enough deterministic evidence to call that candidate supported.
        Embedding similarity can never compensate for a missing contract.
        """
        intent_type = str(entry.get("intent_type") or "").strip()
        qu = self._query_understanding(question)
        constraints = self._constraint_evidence(question, entry)
        required_facets = frozenset(INTENT_REQUIRED_FACETS.get(intent_type, set()))
        matched_facets = frozenset(set(required_facets) & set(qu.facets))

        # Candidate-specific object contract. This is the strongest protection
        # against semantic-neighbour errors such as generic "верификация" being
        # routed to phone/income verification.
        required_objects = REQUIRED_OBJECTS.get(intent_type, frozenset())
        matched_objects = frozenset(set(required_objects) & set(qu.objects)) if required_objects else frozenset(qu.objects)

        positive_hits = self._lexical_phrase_hits(question, entry)
        signal_pos, signal_neg = self._intent_signal_hits(question, entry)
        fuzzy = self._best_fuzzy_phrase_score(question, entry)

        if constraints.get("forbidden_hits"):
            return ContractEvidence(
                status="contradicted",
                reason="forbidden_constraint",
                required_objects=frozenset(required_objects),
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                contradictions=frozenset(constraints["forbidden_hits"]),
                strength=0.0,
            )

        # Explicit KB evidence can prove an intent even when the lightweight
        # facet parser misses a natural inflection. This prevents architecture
        # hardening from rejecting known-good production phrases. Generic
        # semantic similarity still cannot take this escape hatch.
        explicit_evidence = bool(positive_hits or signal_pos)

        if required_facets and not matched_facets and not explicit_evidence:
            return ContractEvidence(
                status="insufficient",
                reason=f"missing_required_intent_facet:{intent_type}",
                required_objects=frozenset(required_objects),
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                missing=required_facets,
                strength=0.0,
            )

        if required_objects and not matched_objects and not explicit_evidence:
            return ContractEvidence(
                status="insufficient",
                reason=f"missing_required_object:{intent_type}",
                required_objects=required_objects,
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                missing=required_objects,
                strength=0.0,
            )

        any_groups = constraints.get("constraint_groups") or []
        if entry.get("constraints") and any_groups:
            failed = [i for i, hits in enumerate(any_groups) if not hits]
            if failed:
                # Explicit must-have-any constraints are hard evidence contracts.
                # If a record is legacy and its list contains very generic words,
                # positive phrase/intent evidence may still rescue it below.
                positive_hits = self._lexical_phrase_hits(question, entry)
                signal_pos, _ = self._intent_signal_hits(question, entry)
                if not positive_hits and not signal_pos:
                    return ContractEvidence(
                        status="insufficient",
                        reason="missing_required_constraint_group",
                        required_objects=frozenset(required_objects),
                        matched_objects=matched_objects,
                        required_facets=required_facets,
                        matched_facets=matched_facets,
                        missing=frozenset(str(i) for i in failed),
                        strength=0.0,
                    )

        # A high-specificity query with a conflicting defining object should not
        # be rescued by a broad family match. This catches cross-domain neighbours.
        candidate_object = str((entry.get("ontology") or self._derive_ontology(entry)).get("object") or "")
        incompatible_pairs = {
            "training": {"membership", "subscription", "rent", "phone", "qr_code", "invite_code"},
            "phone": {"income", "membership", "subscription", "rent", "qr_code", "invite_code"},
            "income": {"phone", "membership", "subscription", "rent", "qr_code", "invite_code"},
            "membership": {"training", "phone", "income", "rent", "subscription"},
        }
        contradictions = set()
        for obj in qu.objects:
            if candidate_object in incompatible_pairs.get(obj, set()):
                contradictions.add(f"object:{obj}")
        if contradictions:
            return ContractEvidence(
                status="contradicted",
                reason="incompatible_object",
                required_objects=frozenset(required_objects),
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                contradictions=frozenset(contradictions),
                strength=0.0,
            )

        # Legacy intents without an explicit contract still need positive
        # evidence. Pure semantic similarity remains insufficient.
        if positive_hits or signal_pos or fuzzy >= 0.90:
            strength = 1.0 if (positive_hits or signal_pos) else 0.75
            return ContractEvidence(
                status="proven", reason="explicit_intent_evidence",
                required_objects=frozenset(required_objects),
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                strength=strength,
            )

        # For contract-bearing intents, absence of defining evidence is an
        # abstention signal. For generic legacy intents we keep status unknown so
        # rare in-domain queries can still reach clarification/LLM, but not an
        # unconditional TEMPLATE.
        if required_objects or required_facets or entry.get("constraints"):
            return ContractEvidence(
                status="insufficient", reason="no_defining_evidence",
                required_objects=frozenset(required_objects),
                matched_objects=matched_objects,
                required_facets=required_facets,
                matched_facets=matched_facets,
                strength=0.0,
            )
        return ContractEvidence(
            status="unknown", reason="legacy_contract_not_declared",
            required_objects=frozenset(), matched_objects=matched_objects,
            required_facets=frozenset(), matched_facets=frozenset(), strength=0.0,
        )

    def _template_eligibility(
        self,
        question: str,
        entry: dict,
        *,
        intent_fit: float,
        family_fit: float,
        fuzzy_score: float = 0.0,
        contrastive_margin: float = 1.0,
        second_is_conflict: bool = False,
        hard_rule: bool = False,
    ) -> tuple[bool, str]:
        """Single final gate before any automatic answer is exposed."""
        structured = self._structured_negation_hits(question, entry)
        if structured:
            return False, "structured_negation:" + ",".join(structured[:2])

        precondition = self._semantic_precondition_reason(question, entry)
        if precondition:
            return False, precondition

        contract = self._intent_contract_evidence(question, entry)
        if contract.status in {"insufficient", "contradicted"}:
            return False, f"intent_contract:{contract.reason}"

        negative_hits = self._negative_hits(question, entry)
        signal_neg = self._intent_signal_hits(question, entry)[1]
        if negative_hits or signal_neg:
            return False, "explicit_candidate_negative"

        if hard_rule:
            passed, reason = self._challenge_hard_rule(question, entry)
            if not passed:
                return False, reason

        # Fuzzy safety is stricter than generic semantic auto-routing. A near
        # duplicate phrase is allowed only when the intent contract and family
        # also agree.
        if fuzzy_score >= 0.98:
            if intent_fit < 0.72 or family_fit < 0.40:
                return False, "fuzzy_safety_failed"

        if second_is_conflict and contrastive_margin < CONTRASTIVE_AUTO_MARGIN:
            return False, "contrastive_margin_too_small"

        # v22 STRICT_TEMPLATE: require stronger fit/family when enabled.
        try:
            from app.core.config import settings as _settings
            strict = bool(getattr(_settings, "STRICT_TEMPLATE", True))
        except Exception:
            strict = True
        if strict:
            if intent_fit < INTENT_FIT_TEMPLATE_MIN:
                return False, "strict_intent_fit"
            if family_fit < MIN_FAMILY_FIT_TEMPLATE:
                return False, "strict_family_fit"
            # Prefer proven contracts for auto-answer under strict mode.
            if contract.status not in {"proven", "unknown"}:
                return False, f"strict_contract:{contract.status}"

        return True, "eligible"

    def _negative_hits(self, question: str, entry: dict) -> list[str]:
        q = normalize_light(question)
        return [
            p for p in entry.get("negative_phrases", [])
            if isinstance(p, str) and normalize_light(p) in q
        ]

    def _required_evidence(self, question: str, entry: dict) -> dict[str, Any]:
        """Treat legacy required_phrases as weak evidence, never as a hard gate.

        Older KB records encode both AND-like and OR-like requirements in one
        flat list. Using them as a hard AND would reject valid Russian variants,
        so we score the strongest matching evidence instead.
        """
        q = normalize_light(question)
        phrases = [
            normalize_light(p) for p in entry.get("required_phrases", [])
            if isinstance(p, str) and normalize_light(p)
        ]
        if not phrases:
            return {"hits": [], "coverage": 0.0}

        hits = [p for p in phrases if p in q]
        coverage = len(hits) / len(phrases)
        return {"hits": hits, "coverage": float(coverage)}

    def _constraint_evidence(self, question: str, entry: dict) -> dict[str, Any]:
        q = normalize_light(question)
        constraints = entry.get("constraints") or {}
        any_groups = constraints.get("must_have_any", [])
        all_groups = constraints.get("must_have_all", [])
        forbidden = constraints.get("must_not_have", [])

        group_hits: list[list[str]] = []
        failed_any = 0
        for group in any_groups:
            group = group if isinstance(group, list) else [group]
            hits = [term for term in group if isinstance(term, str) and normalize_light(term) in q]
            group_hits.append(hits)
            if not hits:
                failed_any += 1

        all_hits = [term for term in all_groups if isinstance(term, str) and normalize_light(term) in q]
        missing_all = [term for term in all_groups if term not in all_hits]
        forbidden_hits = [term for term in forbidden if isinstance(term, str) and normalize_light(term) in q]

        return {
            "constraint_groups": group_hits,
            "failed_any_groups": failed_any,
            "all_hits": all_hits,
            "missing_all": missing_all,
            "forbidden_hits": forbidden_hits,
        }

    @staticmethod
    def _real_intent_evidence(item: dict) -> float:
        """Affirmative intent evidence independent of embedding similarity."""
        positive = 1.0 if item.get("positive_hits") else 0.0
        lexical = float(item.get("lexical_score", 0.0))
        fuzzy = float(item.get("fuzzy_phrase_score", 0.0))
        required = float((item.get("required_evidence") or {}).get("coverage", 0.0))
        intent = float(item.get("intent_fit_score", 0.0))
        family = float(item.get("family_fit_score", 0.0))
        return float(min(1.0, 0.34 * positive + 0.20 * lexical + 0.14 * fuzzy + 0.12 * required + 0.12 * intent + 0.08 * family))

    @staticmethod
    def _candidate_has_direct_support(item: dict) -> bool:
        return bool(item.get("positive_hits") or float(item.get("lexical_score", 0.0)) >= 0.38 or float(item.get("fuzzy_phrase_score", 0.0)) >= 0.90 or float((item.get("required_evidence") or {}).get("coverage", 0.0)) >= 0.50)

    @staticmethod
    def _has_strong_affirmative_evidence(item: dict) -> bool:
        """Decision-layer gate (points 3/6): ranking similarity is not enough.

        TEMPLATE may only fire on strong affirmative evidence — direct phrase hit,
        near-exact fuzzy match, or strong lexical + intent/family agreement.
        """
        positive = bool(item.get("positive_hits"))
        fuzzy = float(item.get("fuzzy_phrase_score", 0.0))
        lexical = float(item.get("lexical_score", 0.0))
        intent = float(item.get("intent_fit_score", 0.0))
        family = float(item.get("family_fit_score", 0.0))
        real_ev = float(item.get("real_intent_evidence", 0.0))

        if item.get("negative_hits") or (item.get("constraint_evidence") or {}).get("forbidden_hits"):
            return False

        # Direct positive phrase hit is the strongest signal.
        if positive and intent >= 0.55 and family >= 0.30 and real_ev >= 0.36:
            return True
        # Near-duplicate wording.
        if fuzzy >= 0.92 and intent >= 0.58 and family >= 0.35 and real_ev >= 0.38:
            return True
        # Strong lexical + intent contract, no contradiction.
        if lexical >= 0.55 and intent >= 0.68 and family >= 0.45 and real_ev >= 0.42:
            return True
        return False

    def _intent_signal_hits(self, question: str, entry: dict) -> tuple[list[str], list[str]]:
        """Return explicit positive/negative intent-signal hits.

        These signals are the KB's most useful supervised-ish features for
        distinguishing categories that share the same nouns (school, mentor,
        chat, payment) but differ by the requested action/state.
        """
        q = normalize_light(question)
        signals = entry.get("intent_signals") or {}
        positive = [
            normalize_light(str(x))
            for x in signals.get("positive", [])
            if isinstance(x, str) and normalize_light(str(x))
        ]
        negative = [
            normalize_light(str(x))
            for x in signals.get("negative", [])
            if isinstance(x, str) and normalize_light(str(x))
        ]
        return (
            [x for x in positive if x in q],
            [x for x in negative if x in q],
        )

    def _pairwise_discriminator(self, question: str, candidate: dict, competitor: dict) -> dict[str, Any]:
        """Compare two close intents using explicit intent contracts.

        Embedding similarity is intentionally excluded here. This stage answers
        the harder question: *which intent better explains the exact action,
        state and constraints expressed by the user?*
        """
        a = candidate["entry"]
        b = competitor["entry"]
        a_pos, a_neg = self._intent_signal_hits(question, a)
        b_pos, b_neg = self._intent_signal_hits(question, b)

        a_ontology = candidate.get("ontology") or self._derive_ontology(a)
        b_ontology = competitor.get("ontology") or self._derive_ontology(b)
        qo = self._query_ontology(question)

        a_supported = set(a.get("intent_facets", [])) or self._entry_supported_facets(a)
        b_supported = set(b.get("intent_facets", [])) or self._entry_supported_facets(b)
        q_facets = self._question_facets(question)

        a_score = 0.0
        b_score = 0.0
        reasons_a: list[str] = []
        reasons_b: list[str] = []

        # Explicit intent signal is the strongest pairwise evidence.
        if a_pos:
            a_score += CONTRASTIVE_SIGNAL_BONUS * min(2, len(a_pos))
            reasons_a.append("positive_signal")
        if b_pos:
            b_score += CONTRASTIVE_SIGNAL_BONUS * min(2, len(b_pos))
            reasons_b.append("positive_signal")
        if a_neg:
            a_score -= CONTRASTIVE_NEGATIVE_BONUS * min(2, len(a_neg))
            reasons_a.append("negative_signal")
        if b_neg:
            b_score -= CONTRASTIVE_NEGATIVE_BONUS * min(2, len(b_neg))
            reasons_b.append("negative_signal")

        # Query facets are much more informative when comparing known neighbors.
        if q_facets:
            overlap_a = q_facets & a_supported
            overlap_b = q_facets & b_supported
            if overlap_a:
                a_score += CONTRASTIVE_FACET_BONUS * min(1.0, len(overlap_a) / max(1, len(q_facets)))
                reasons_a.append("facet_match")
            if overlap_b:
                b_score += CONTRASTIVE_FACET_BONUS * min(1.0, len(overlap_b) / max(1, len(q_facets)))
                reasons_b.append("facet_match")

        # Exact ontology dimensions give a second independent axis.
        for dim, weight in (("object", 0.05), ("action", 0.025), ("state", 0.025), ("facet", 0.02)):
            qv = qo.get(dim) or set()
            if not qv:
                continue
            av = a_ontology.get(dim)
            bv = b_ontology.get(dim)
            if av in qv:
                a_score += min(weight, CONTRASTIVE_ONTOLOGY_BONUS)
            elif av:
                a_score -= min(weight, CONTRASTIVE_ONTOLOGY_BONUS)
            if bv in qv:
                b_score += min(weight, CONTRASTIVE_ONTOLOGY_BONUS)
            elif bv:
                b_score -= min(weight, CONTRASTIVE_ONTOLOGY_BONUS)

        # Required/forbidden constraints are strong discriminators, especially
        # for registration vs registration-failure and Telegram link vs membership.
        a_req = candidate.get("constraint_evidence") or {}
        b_req = competitor.get("constraint_evidence") or {}
        if a_req.get("forbidden_hits"):
            a_score -= 0.22
            reasons_a.append("forbidden")
        if b_req.get("forbidden_hits"):
            b_score -= 0.22
            reasons_b.append("forbidden")
        a_failed = int(a_req.get("failed_any_groups", 0))
        b_failed = int(b_req.get("failed_any_groups", 0))
        if a_failed:
            a_score -= min(0.18, 0.06 * a_failed)
        if b_failed:
            b_score -= min(0.18, 0.06 * b_failed)

        # Direct phrase evidence should win over generic semantic closeness.
        if candidate.get("positive_hits"):
            a_score += 0.06
            reasons_a.append("direct_phrase")
        if competitor.get("positive_hits"):
            b_score += 0.06
            reasons_b.append("direct_phrase")

        advantage = float(a_score - b_score)
        return {
            "advantage": advantage,
            "candidate_evidence": float(a_score),
            "competitor_evidence": float(b_score),
            "candidate_reasons": reasons_a,
            "competitor_reasons": reasons_b,
        }

    def _contrastive_context_ids(self, ranked: list[dict]) -> list[str]:
        """Expand LLM/choice context with declared confusion-neighbors.

        Retrieval can miss the correct neighbor when several categories share
        vocabulary. The KB conflict graph is therefore used as a recall safety
        net for the final decision layer.
        """
        ids: list[str] = []
        seen: set[str] = set()
        for item in ranked[:3]:
            entry = item["entry"]
            for raw_id in [entry.get("id"), *(entry.get("conflict_ids", []) or [])]:
                kb_id = str(raw_id)
                if not kb_id or kb_id == "default" or kb_id in seen:
                    continue
                if kb_id in self.entries_by_id:
                    ids.append(kb_id)
                    seen.add(kb_id)
                if len(ids) >= CONTRASTIVE_MAX_CONTEXT_NEIGHBORS:
                    return ids
        return ids

    def _apply_contrastive_scores(self, question: str, ranked: list[dict]) -> list[dict]:
        if len(ranked) < 2:
            for item in ranked:
                item["contrastive_margin"] = 1.0
                item["contrastive_competitor_id"] = None
                item["contrastive_reason"] = "single_candidate"
            return ranked

        # Freeze the retrieval score before applying contrastive adjustments so
        # the result is order-independent.
        base_scores = {
            str(item["entry"]["id"]): float(item.get("final_score", 0.0))
            for item in ranked
        }
        pair_updates: list[tuple[dict, dict]] = []

        for item in ranked:
            item["contrastive_margin"] = 1.0
            item["contrastive_competitor_id"] = None
            item["contrastive_reason"] = "no_close_competitor"

        for i, item in enumerate(ranked):
            entry_id = str(item["entry"]["id"])
            conflict_ids = {str(x) for x in item["entry"].get("conflict_ids", [])}
            best_competitor = None
            best_key = (-1.0, -1.0)
            for j, other in enumerate(ranked):
                if i == j:
                    continue
                other_id = str(other["entry"]["id"])
                same_family = str(other["entry"].get("family") or "") == str(item["entry"].get("family") or "")
                is_conflict = other_id in conflict_ids or entry_id in {str(x) for x in other["entry"].get("conflict_ids", [])}
                if not (is_conflict or same_family):
                    continue
                score = base_scores.get(other_id, 0.0)
                priority = 1.0 if is_conflict else 0.0
                key = (priority, score)
                if best_competitor is None or key > best_key:
                    best_competitor = other
                    best_key = key
            if best_competitor is None or base_scores.get(str(best_competitor["entry"]["id"]), 0.0) < CONTRASTIVE_MIN_COMPETITOR_SCORE:
                continue

            pair = self._pairwise_discriminator(question, item, best_competitor)
            item["contrastive_margin"] = float(pair["advantage"])
            item["contrastive_competitor_id"] = str(best_competitor["entry"]["id"])
            reasons = []
            if pair["candidate_reasons"]:
                reasons.append("A:" + ",".join(pair["candidate_reasons"][:4]))
            if pair["competitor_reasons"]:
                reasons.append("B:" + ",".join(pair["competitor_reasons"][:4]))
            item["contrastive_reason"] = ";".join(reasons) or "neutral"
            pair_updates.append((item, pair))

        for item, pair in pair_updates:
            base = base_scores[str(item["entry"]["id"])]
            item["ranking_score"] = float(max(0.0, min(1.0, item.get("ranking_score", base) + 0.10 * pair["advantage"])))
            item["final_score"] = float(max(0.0, min(1.0, base + 0.06 * pair["advantage"])))

        ranked.sort(
            key=lambda x: (
                x.get("ranking_score", 0.0),
                x.get("final_score", 0.0),
                x.get("contrastive_margin", 0.0),
                x.get("intent_fit_score", 0.0),
            ),
            reverse=True,
        )
        return ranked


    def _rank_candidates(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        query_quality = _query_quality(question)
        pairs = []
        for item in candidates:
            entry = item["entry"]
            pairs.append((question, self._reranker_text(entry, item["best_phrase"])))

        generic_scores = [0.0] * len(candidates)
        if self.reranker is not None:
            raw_scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            generic_scores = [
                self._sigmoid(float(np.asarray(x).reshape(-1)[0]))
                for x in raw_scores
            ]

        ranked: list[dict] = []
        for item, generic in zip(candidates, generic_scores):
            entry = item["entry"]
            positives = self._lexical_phrase_hits(question, entry)
            negatives = self._negative_hits(question, entry)
            constraints = self._constraint_evidence(question, entry)
            required = self._required_evidence(question, entry)
            fuzzy_phrase = self._best_fuzzy_phrase_score(question, entry)
            intent_fit, intent_fit_reason = self._intent_fit(
                question,
                entry,
                fuzzy_phrase=fuzzy_phrase,
            )
            family_fit = self._family_fit(question, entry)
            ontology_alignment, ontology_reason = self._ontology_alignment(question, entry)
            domain_support = self._domain_support(question, entry)

            semantic = (
                SEMANTIC_MAX_WEIGHT * item["embedding_score"]
                + SEMANTIC_TOP3_WEIGHT * item["semantic_top3"]
                + SEMANTIC_CATEGORY_WEIGHT * item["semantic_prototype"]
            )
            negative_semantic = float(item.get("negative_semantic_score", 0.0))
            hard_negative_gap = float(item["embedding_score"] - negative_semantic)

            # Convert BM25 to a stable bounded signal. Do NOT divide by the
            # maximum candidate score: that makes a weak query look "0.95"
            # merely because one category happened to be the best of a bad set.
            bm25_raw = max(0.0, float(item["bm25_raw"]))
            lexical_score = 1.0 - math.exp(-bm25_raw / 1.6)

            rrf_max = 2.0 / (RRF_K + 1.0)
            rrf_norm = min(1.0, float(item["rrf_score"]) / max(rrf_max, 1e-9))

            phrase_evidence = 1.0 if positives else fuzzy_phrase
            constraint_coverage = float(required["coverage"])
            contradiction = bool(
                negatives or constraints.get("forbidden_hits")
            )

            # Evidence quality: direct phrase/required evidence helps, while
            # contradictions are handled as a hard penalty below.
            evidence = (
                0.55 * phrase_evidence
                + 0.25 * constraint_coverage
                + 0.20 * family_fit
            )

            base_weights = {
                "semantic": SEMANTIC_MAX_WEIGHT + SEMANTIC_TOP3_WEIGHT + SEMANTIC_CATEGORY_WEIGHT,
                "lexical": LEXICAL_SCORE_WEIGHT,
                "fuzzy": FUZZY_SCORE_WEIGHT,
                "intent": INTENT_FIT_WEIGHT,
                "family": FAMILY_FIT_WEIGHT,
                "rrf": RRF_WEIGHT,
                "rerank": RERANK_WEIGHT if self.reranker is not None else 0.0,
                "evidence": 0.08,
                # Structured compatibility is deliberately meaningful but not
                # dominant: it protects against false semantic matches while
                # allowing terse queries with no explicit ontology to survive.
                "ontology": 0.10,
            }
            total_weight = sum(base_weights.values())
            scale = 1.0 / max(total_weight, 1e-9)

            score = (
                base_weights["semantic"] * semantic
                + base_weights["lexical"] * lexical_score
                + base_weights["fuzzy"] * fuzzy_phrase
                + base_weights["intent"] * intent_fit
                + base_weights["family"] * family_fit
                + base_weights["rrf"] * rrf_norm
                + base_weights["rerank"] * generic
                + base_weights["evidence"] * evidence
                + base_weights["ontology"] * ontology_alignment
            ) * scale

            if contradiction:
                score -= 0.20
            # Semantic hard-negative evidence: a candidate that resembles its own
            # negative examples almost as much as its positives is unsafe.
            negative_contrast = float(item["embedding_score"] - negative_semantic) if negative_semantic > 0.0 else 1.0
            if negative_semantic > 0.0:
                # Penalize candidates whose own negative examples are as close as
                # or closer than their positives. This is a contradiction signal,
                # not a generic score penalty.
                score -= NEGATIVE_SEMANTIC_WEIGHT * max(0.0, -negative_contrast + NEGATIVE_CONTRAST_MIN)

            qo = self._query_ontology(question)
            ontology = entry.get("ontology") or self._derive_ontology(entry)
            if ontology_alignment <= ONTOLOGY_HARD_MISMATCH and qo["object"]:
                score -= OBJECT_MISMATCH_PENALTY
            elif ontology_alignment < ONTOLOGY_ALIGNMENT_MIN and any(qo.get(k) for k in ("action", "state", "facet")):
                score -= 0.08

            # Query quality is deliberately NOT used as a ranking cap. A short
            # but valid intent can have almost no question-mark/action-form
            # signal (e.g. "Хотел бы стать наставником"). Noise/OOS handling is
            # performed by the decision layer, while evidence and contradictions
            # determine whether a candidate may actually be selected.

            contract = self._intent_contract_evidence(question, entry)
            if contract.status == "contradicted":
                score -= 0.28
            elif contract.status == "insufficient":
                score -= 0.18

            final_score = float(max(0.0, min(1.0, score)))

            # Ranking score is deliberately distinct from decision score:
            # intent fit and family fit can reorder candidates, but the decision
            # gates below use the bounded final confidence.
            ranking_score = float(
                0.78 * final_score
                + 0.14 * intent_fit
                + 0.08 * family_fit
            )

            ranked_item = dict(item)
            ranked_item.update({
                "rerank_score": generic,
                "semantic_score": float(semantic),
                "negative_semantic_score": negative_semantic,
                "hard_negative_gap": hard_negative_gap,
                "ontology": ontology,
                "query_ontology": qo,
                "ontology_alignment_score": float(ontology_alignment),
                "ontology_alignment_reason": ontology_reason,
                "domain_support_score": float(domain_support),
                "negative_contrast_score": float(negative_contrast),
                "rrf_normalized": float(rrf_norm),
                "lexical_score": float(lexical_score),
                "fuzzy_phrase_score": float(fuzzy_phrase),
                "required_evidence": required,
                "intent_fit_score": float(intent_fit),
                "intent_fit_reason": intent_fit_reason,
                "family_fit_score": float(family_fit),
                "query_quality_score": float(query_quality.score),
                "query_quality_reason": query_quality.reason,
                "query_quality": query_quality,
                "evidence_score": float(evidence),
                "ranking_score": ranking_score,
                "final_score": final_score,
                "positive_hits": positives,
                "negative_hits": negatives,
                "constraint_evidence": constraints,
            })
            contract = self._intent_contract_evidence(question, entry)
            ranked_item["intent_contract_status"] = contract.status
            ranked_item["intent_contract_reason"] = contract.reason
            ranked_item["intent_contract_strength"] = contract.strength
            ranked_item["intent_contract_required_objects"] = sorted(contract.required_objects)
            ranked_item["intent_contract_matched_objects"] = sorted(contract.matched_objects)
            ranked_item["intent_contract_required_facets"] = sorted(contract.required_facets)
            ranked_item["intent_contract_matched_facets"] = sorted(contract.matched_facets)
            ranked_item["real_intent_evidence"] = self._real_intent_evidence(ranked_item)
            ranked_item["direct_support"] = self._candidate_has_direct_support(ranked_item)
            ranked.append(ranked_item)

        ranked.sort(
            key=lambda x: (
                x.get("ranking_score", 0.0),
                x.get("final_score", 0.0),
                x.get("intent_fit_score", 0.0),
            ),
            reverse=True,
        )
        return self._apply_contrastive_scores(question, ranked[:FINAL_CANDIDATES])

    @staticmethod
    def _reranker_text(entry: dict, best_phrase: str) -> str:
        category = str(entry.get("category") or "")
        family = str(entry.get("family") or "")
        intent_type = str(entry.get("intent_type") or "")
        facets = ", ".join(str(x) for x in entry.get("intent_facets", [])[:8])
        ontology = entry.get("ontology") or self._derive_ontology(entry)
        definition = str(entry.get("definition") or entry.get("scope") or entry.get("description") or "")
        positives = "; ".join(str(x) for x in entry.get("positive_examples", [])[:8])
        negatives = "; ".join(str(x) for x in entry.get("negative_phrases", [])[:5])
        signal_pos = "; ".join(str(x) for x in (entry.get("intent_signals") or {}).get("positive", [])[:8])
        signal_neg = "; ".join(str(x) for x in (entry.get("intent_signals") or {}).get("negative", [])[:8])
        return (
            f"Семейство: {family}. Объект: {ontology.get('object')}. Действие: {ontology.get('action')}. Состояние: {ontology.get('state')}. Фасет: {ontology.get('facet')}. Тип интента: {intent_type}. Фасеты: {facets}. "
            f"Категория: {category}. Определение: {definition}. "
            f"Положительные примеры: {positives}. "
            f"Intent-сигналы: {signal_pos}. Не подходит для: {signal_neg}. "
            f"Не использовать для: {negatives}. Лучший пример: {best_phrase}."
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    # --------------------------------------------------------
    # OUTPUT / LLM CONTEXT
    # --------------------------------------------------------

    @staticmethod
    def _clean_entry_for_llm(entry: dict) -> dict:
        allowed = {
            "id", "category", "family", "definition", "match_phrases", "positive_examples",
            "negative_phrases", "required_phrases", "constraints", "priority", "scope",
            "when_to_use", "when_not_to_use", "confused_with", "requires_clarification",
            "clarification_question", "template_answer", "description", "hard_negative_examples",
            "intent_type", "intent_facets", "intent_signals", "choice_label",
        }
        return {k: v for k, v in entry.items() if k in allowed}

    def _build_relevant_entries(self, ranked: list[dict]) -> list[dict]:
        """Build the LLM decision context from ranked candidates + contrastive neighbors."""
        if not ranked:
            return []

        selected_ids = self._contrastive_context_ids(ranked)
        ranked_by_id = {str(item["entry"]["id"]): item for item in ranked}

        # Include all scored neighbors first, then unscored conflict-neighbors
        # as context-only records. The LLM may select only valid KB ids.
        ordered_ids: list[str] = []
        for item in ranked[:LLM_TOP_K]:
            kb_id = str(item["entry"]["id"])
            if kb_id not in ordered_ids:
                ordered_ids.append(kb_id)
        for kb_id in selected_ids:
            if kb_id not in ordered_ids:
                ordered_ids.append(kb_id)

        result = []
        for kb_id in ordered_ids[: max(LLM_TOP_K, CONTRASTIVE_MAX_CONTEXT_NEIGHBORS)]:
            item = ranked_by_id.get(kb_id)
            if item is None:
                entry = self.entries_by_id.get(kb_id)
                if not entry:
                    continue
                clean = self._clean_entry_for_llm(entry)
                clean.update({
                    "_context_only": True,
                    "_contrastive_neighbor": True,
                    "_embedding_score": 0.0,
                    "_semantic_top3": 0.0,
                    "_semantic_prototype": 0.0,
                    "_bm25_score": 0.0,
                    "_rrf_score": 0.0,
                    "_lexical_score": 0.0,
                    "_rerank_score": 0.0,
                    "_semantic_score": 0.0,
                    "_negative_semantic_score": 0.0,
                    "_hard_negative_gap": 0.0,
                    "_ontology": entry.get("ontology", self._derive_ontology(entry)),
                    "_query_ontology": {},
                    "_ontology_alignment_score": 0.0,
                    "_ontology_alignment_reason": "contrastive_context_only",
                    "_domain_support_score": 0.0,
                    "_negative_contrast_score": 0.0,
                    "_rrf_normalized": 0.0,
                    "_family_fit_score": 0.0,
                    "_query_quality_score": 0.0,
                    "_query_quality_reason": "contrastive_context_only",
                    "_evidence_score": 0.0,
                    "_fuzzy_phrase_score": 0.0,
                    "_required_evidence": {},
                    "_intent_fit_score": 0.0,
                    "_intent_fit_reason": "contrastive_context_only",
                    "_ranking_score": 0.0,
                    "_final_score": 0.0,
                    "_best_match_phrase": "",
                    "_positive_hits": [],
                    "_negative_hits": [],
                    "_constraint_evidence": {},
                    "_intent_type": entry.get("intent_type"),
                    "_intent_facets": entry.get("intent_facets", []),
                    "_choice_label": entry.get("choice_label"),
                })
                result.append(clean)
                continue

            clean = self._clean_entry_for_llm(item["entry"])
            clean.update({
                "_context_only": False,
                "_contrastive_neighbor": str(item.get("contrastive_competitor_id") or "") in ordered_ids,
                "_embedding_score": float(item["embedding_score"]),
                "_semantic_top3": float(item["semantic_top3"]),
                "_semantic_prototype": float(item["semantic_prototype"]),
                "_bm25_score": float(item["bm25_score"]),
                "_rrf_score": float(item["rrf_score"]),
                "_lexical_score": float(item.get("lexical_score", 0.0)),
                "_rerank_score": float(item["rerank_score"]),
                "_semantic_score": float(item["semantic_score"]),
                "_negative_semantic_score": float(item.get("negative_semantic_score", 0.0)),
                "_hard_negative_gap": float(item.get("hard_negative_gap", 0.0)),
                "_ontology": item.get("ontology", {}),
                "_query_ontology": {k: sorted(v) for k, v in (item.get("query_ontology") or {}).items()},
                "_ontology_alignment_score": float(item.get("ontology_alignment_score", 0.0)),
                "_ontology_alignment_reason": item.get("ontology_alignment_reason", ""),
                "_domain_support_score": float(item.get("domain_support_score", 0.0)),
                "_negative_contrast_score": float(item.get("negative_contrast_score", 0.0)),
                "_rrf_normalized": float(item.get("rrf_normalized", 0.0)),
                "_family_fit_score": float(item.get("family_fit_score", 0.0)),
                "_query_quality_score": float(item.get("query_quality_score", 0.0)),
                "_query_quality_reason": item.get("query_quality_reason", ""),
                "_evidence_score": float(item.get("evidence_score", 0.0)),
                "_fuzzy_phrase_score": float(item.get("fuzzy_phrase_score", 0.0)),
                "_required_evidence": item.get("required_evidence", {}),
                "_intent_fit_score": float(item.get("intent_fit_score", 0.0)),
                "_intent_fit_reason": item.get("intent_fit_reason", ""),
                "_ranking_score": float(item.get("ranking_score", item["final_score"])),
                "_final_score": float(item["final_score"]),
                "_best_match_phrase": item["best_phrase"],
                "_positive_hits": item["positive_hits"],
                "_negative_hits": item["negative_hits"],
                "_constraint_evidence": item["constraint_evidence"],
                "_contrastive_margin": float(item.get("contrastive_margin", 0.0)),
                "_contrastive_competitor_id": item.get("contrastive_competitor_id"),
                "_contrastive_reason": item.get("contrastive_reason", ""),
                "_intent_type": item["entry"].get("intent_type"),
                "_intent_facets": item["entry"].get("intent_facets", []),
                "_choice_label": item["entry"].get("choice_label"),
            })
            result.append(clean)
        return result

    # --------------------------------------------------------
    # USER-FACING FALLBACK CANDIDATES
    # --------------------------------------------------------

    @staticmethod
    def _choice_candidate_payload(item: dict) -> dict:
        entry = item["entry"]
        return {
            "kb_id": str(entry["id"]),
            "choice_label": str(entry.get("choice_label") or entry.get("category") or entry["id"]),
            "category": str(entry.get("category") or ""),
            "score": float(item.get("final_score", 0.0)),
            "ranking_score": float(item.get("ranking_score", item.get("final_score", 0.0))),
            "second_score": 0.0,
            "semantic_score": float(item.get("semantic_score", 0.0)),
            "lexical_score": float(item.get("lexical_score", 0.0)),
            "fuzzy_score": float(item.get("fuzzy_phrase_score", 0.0)),
            "evidence_score": float(item.get("evidence_score", 0.0)),
            "real_intent_evidence": float(item.get("real_intent_evidence", 0.0)),
            "margin": 0.0,
            "embedding_score": float(item.get("embedding_score", 0.0)),
            "intent_fit_score": float(item.get("intent_fit_score", 0.0)),
            "intent_fit_reason": str(item.get("intent_fit_reason", "")),
            "family_fit_score": float(item.get("family_fit_score", 0.0)),
            "query_quality_score": float(item.get("query_quality_score", 0.0)),
            "evidence_score": float(item.get("evidence_score", 0.0)),
            "negative_semantic_score": float(item.get("negative_semantic_score", 0.0)),
            "hard_negative_gap": float(item.get("hard_negative_gap", 0.0)),
            "negative_contrast_score": float(item.get("negative_contrast_score", 0.0)),
            "ontology_alignment_score": float(item.get("ontology_alignment_score", 0.0)),
            "domain_support_score": float(item.get("domain_support_score", 0.0)),
            "ontology": item.get("ontology", {}),
        }

    def _force_top_choice_candidates(self, ranked: list[dict], n: int = 3) -> list[dict]:
        """TEMP: always take top-N ranked as choice candidates (no safety filters).
        Used while LLM path is disabled.
        """
        if not ranked:
            return []
        top_score = float(ranked[0].get("final_score", 0.0))
        second_score = float(ranked[1].get("final_score", 0.0)) if len(ranked) > 1 else 0.0
        top_margin = top_score - second_score if len(ranked) > 1 else top_score
        result: list[dict] = []
        for item in ranked[:n]:
            payload = self._choice_candidate_payload(item)
            payload["margin"] = float(top_margin)
            result.append(payload)
        return result

    def _category_choice_candidates(self, ranked: list[dict]) -> list[dict]:
        """Primary middle mode (v22.1): 2–3 alternatives when TEMPLATE is not safe.

        Filters are much milder than TEMPLATE. Only hard negatives/contradictions
        are dropped so mid-confidence queries get a menu instead of rephrase.
        """
        if not ranked:
            return []

        top = ranked[0]
        top_score = float(top.get("final_score", 0.0))
        second_score = float(ranked[1].get("final_score", 0.0)) if len(ranked) > 1 else 0.0
        top_margin = top_score - second_score if len(ranked) > 1 else top_score
        quality = float(top.get("query_quality_score", 0.0))

        if quality < MIN_QUERY_QUALITY_CHOICE and top_score < 0.55:
            return []
        if top_score < CATEGORY_CHOICE_MIN_SCORE:
            return []
        # Huge margin alone does not block Top-3: TEMPLATE still needs AUTO_SCORE.
        # Only skip choice when margin is extreme AND top is already template-safe.
        if top_margin > CATEGORY_CHOICE_MAX_MARGIN and top_score >= AUTO_SCORE:
            return []

        eligible: list[dict] = []
        for item in ranked[: max(CATEGORY_CHOICE_MAX_CANDIDATES + 3, 6)]:
            score = float(item.get("final_score", 0.0))
            fit = float(item.get("intent_fit_score", 0.0))
            constraints = item.get("constraint_evidence", {}) or {}
            contract_status = str(item.get("intent_contract_status", "unknown"))

            # Hard blocks only.
            if contract_status == "contradicted":
                continue
            if item.get("negative_hits"):
                continue
            if constraints.get("forbidden_hits"):
                continue
            if score < CATEGORY_CHOICE_MIN_SCORE * 0.85 and item is not ranked[0]:
                # keep top even if slightly under threshold when building menu
                if float(item.get("final_score", 0.0)) < 0.30:
                    continue
            if score < 0.30:
                continue
            if fit < CATEGORY_CHOICE_MIN_INTENT_FIT and score < 0.50:
                continue
            if float(item.get("real_intent_evidence", 0.0)) < CATEGORY_CHOICE_MIN_REAL_EVIDENCE and not item.get("direct_support"):
                if score < 0.45:
                    continue
            if CATEGORY_CHOICE_REQUIRE_DIRECT_SUPPORT and not item.get("direct_support"):
                continue
            if top_score - score > max(0.45, CATEGORY_CHOICE_MIN_SCORE_GAP_FROM_BEST * 4.0):
                continue

            payload = self._choice_candidate_payload(item)
            payload["margin"] = float(top_margin)
            eligible.append(payload)

        # Fallback: if strict filters left <2, take top-3 ranked with any score >= 0.28
        if len(eligible) < CATEGORY_CHOICE_MIN_CANDIDATES:
            eligible = []
            for item in ranked[:CATEGORY_CHOICE_MAX_CANDIDATES + 1]:
                if item.get("negative_hits"):
                    continue
                if (item.get("constraint_evidence") or {}).get("forbidden_hits"):
                    continue
                if float(item.get("final_score", 0.0)) < 0.28:
                    continue
                payload = self._choice_candidate_payload(item)
                payload["margin"] = float(top_margin)
                eligible.append(payload)
                if len(eligible) >= CATEGORY_CHOICE_MAX_CANDIDATES:
                    break

        if len(eligible) < CATEGORY_CHOICE_MIN_CANDIDATES:
            return []

        unique: list[dict] = []
        seen_ids: set[str] = set()
        seen_intent_types: set[str] = set()
        for candidate in eligible:
            kb_id = candidate["kb_id"]
            entry = self.entries_by_id.get(kb_id)
            intent_type = str(entry.get("intent_type") or kb_id) if entry else kb_id
            if kb_id in seen_ids:
                continue
            if intent_type in seen_intent_types and len(unique) >= CATEGORY_CHOICE_MIN_CANDIDATES:
                continue
            unique.append(candidate)
            seen_ids.add(kb_id)
            seen_intent_types.add(intent_type)

        return unique[:CATEGORY_CHOICE_MAX_CANDIDATES]

    def get_category_choice_candidates(
        self,
        question: str,
        top_n: int = CATEGORY_CHOICE_MAX_CANDIDATES,
    ) -> list[dict]:
        """Get safe candidates for the future CATEGORY_CHOICE conversation state.

        This method does not change the normal match() routing yet. It is a
        pure decision helper used by the orchestration/state layer.
        """
        if top_n < CATEGORY_CHOICE_MIN_CANDIDATES:
            raise ValueError("top_n must be at least 2")

        candidates = self._retrieve_candidates(
            question,
            max(RETRIEVAL_TOP_K, top_n),
        )
        ranked = self._rank_candidates(question, candidates)
        eligible = self._category_choice_candidates(ranked)
        return eligible[:top_n]

    @staticmethod
    def format_category_choice(
        candidates: list[dict],
        *,
        intro: str = (
            "Если среди представленных ниже вариантов есть тот, "
            "который относится к вашему вопросу, напишите его номер:"
        ),
    ) -> str:
        """Render a user-facing category choice message.

        Only presentation labels are used here. KB IDs/scores stay internal.
        """
        if not candidates:
            raise ValueError("At least one candidate is required")

        lines = [intro, ""]
        for index, candidate in enumerate(candidates, start=1):
            lines.append(f"{index}. {candidate['choice_label']}")
        return "\n".join(lines)

    @staticmethod
    def parse_category_choice(
        text: str,
        *,
        max_index: int,
    ) -> list[int]:
        """Parse natural user selections while a CATEGORY_CHOICE is pending.

        Supported examples: ``1``, ``1 и 3``, ``вариант 2``,
        ``мне подходит первый``, ``беру первый и третий``.
        Numeric parsing remains bounded by max_index, so dates/years cannot
        accidentally become a valid choice.
        """
        if max_index < 1:
            raise ValueError("max_index must be positive")

        normalized = normalize_light(text)
        if not normalized:
            return []

        ordinals = {
            "первый": 1, "первая": 1, "первое": 1,
            "второй": 2, "вторая": 2, "второе": 2,
            "третий": 3, "третья": 3, "третье": 3,
        }
        values: list[int] = []

        # First parse explicit Russian ordinals.
        for token in normalized.split():
            value = ordinals.get(token.strip(".,!?;:"))
            if value is not None and value <= max_index and value not in values:
                values.append(value)

        # Then parse explicit numeric choices. Reject out-of-range numbers so
        # messages such as "поток 2026" never turn into a choice accidentally.
        for raw in re.findall(r"(?<!\w)([1-9]\d*)(?!\w)", normalized):
            value = int(raw)
            if value < 1 or value > max_index:
                return []
            if value not in values:
                values.append(value)

        return values

    @staticmethod
    def is_rephrase_candidate(result: MatchResult) -> bool:
        """Whether a result is weak enough for the REPHRASE state."""
        if result.mode != MatchMode.NO_MATCH:
            return False
        return (
            result.score < REPHRASE_SCORE_MAX
            or result.intent_fit_score <= REPHRASE_INTENT_FIT_MAX
            or getattr(result, "query_quality_score", 0.0) <= REPHRASE_QUERY_QUALITY_MAX
        )

    # --------------------------------------------------------
    # MAIN MATCH
    # --------------------------------------------------------

    def match(self, question: str) -> MatchResult:
        """Classify one user message with an evidence-first policy.

        Key design rules:
        * deterministic high-precision rules run BEFORE generic quality checks;
        * quality is a noise signal, never a blanket "not a question" gate;
        * embeddings generate candidates, they never prove an intent alone;
        * automatic answers require either deterministic evidence, direct KB
          evidence, or strong semantic+intent agreement;
        * ambiguous/unsupported requests abstain instead of inventing a category.
        """
        question = (question or "").strip()

        if not question:
            return MatchResult(
                MatchMode.CLARIFICATION,
                0.0,
                None,
                self.default_answer,
                None,
                [],
                subquestion=question,
                decision_reason="empty_question",
            )

        if contains_operator_request(question):
            return MatchResult(
                MatchMode.OPERATOR,
                1.0,
                None,
                None,
                None,
                [],
                subquestion=question,
                decision_reason="user_requested_operator",
            )

        # High-risk: refund / personal data / complaint — never TEMPLATE.
        high_risk = detect_high_risk(question)
        if high_risk:
            logger.info(
                "high_risk_detected | kind=%s | question=%s",
                high_risk,
                question[:300],
            )
            return MatchResult(
                MatchMode.NO_MATCH,
                1.0,
                None,
                None,
                None,
                [],
                subquestion=question,
                decision_reason=f"high_risk:{high_risk}",
            )

        if not self.entries:
            return MatchResult(
                MatchMode.NO_MATCH, 0.0, None, None, None, [],
                subquestion=question,
                decision_reason="empty_knowledge_base",
            )

        # 1. Deterministic business rules have the highest precision and must
        # run before the generic query-quality filter. This fixes valid short
        # inputs such as "Хочу стать наставником".
        hard_id = self._hard_rule_match(question)
        quality = _query_quality(question)
        if hard_id:
            entry = self.entries_by_id.get(hard_id)
            if entry and not self._negative_hits(question, entry):
                # A deterministic rule is a high-value hypothesis, not an
                # unconditional classification. It must still agree with the
                # extracted intent facets / ontology. This prevents broad
                # topic rules (e.g. any mention of “наставник”) from sending a
                # wrong template with score=0.995.
                hard_fuzzy = self._best_fuzzy_phrase_score(question, entry)
                fit, fit_reason = self._intent_fit(question, entry, fuzzy_phrase=hard_fuzzy)
                family_fit = self._family_fit(question, entry)
                alignment, alignment_reason = self._ontology_alignment(question, entry)
                q_facets = self._question_facets(question)
                required_facets = INTENT_REQUIRED_FACETS.get(str(entry.get("intent_type") or ""), set())
                explicit_facet_ok = (not required_facets) or bool(q_facets & required_facets)
                hard_rule_type = str(entry.get("intent_type") or "")
                explicit_positive = bool(
                    self._lexical_phrase_hits(question, entry)
                    or self._intent_signal_hits(question, entry)[0]
                    or (q_facets & required_facets)
                    or hard_rule_type in {"flow_date", "school_started"}
                )
                hard_safe = (
                    fit >= (0.80 if required_facets else 0.72)
                    and family_fit >= 0.45
                    and alignment >= (0.65 if required_facets else 0.55)
                    and explicit_facet_ok
                    and explicit_positive
                )
                if hard_safe:
                    hard_safe, hard_reason = self._template_eligibility(
                        question, entry, intent_fit=fit, family_fit=family_fit,
                        fuzzy_score=hard_fuzzy, hard_rule=True,
                    )
                    if not hard_safe:
                        logger.debug("hard_rule veto | kb_id=%s | reason=%s", hard_id, hard_reason)
                logger.debug(
                    "hard_rule hypothesis | kb_id=%s | fit=%.3f | family=%.3f | alignment=%.3f | facets=%s | safe=%s | reason=%s",
                    hard_id, fit, family_fit, alignment, sorted(q_facets), hard_safe, fit_reason,
                )
                if not hard_safe:
                    # Do not trust the hard rule; continue through the full
                    # hybrid classifier where competing intents are ranked and
                    # compared contrastively.
                    hard_id = None
                else:
                    if entry.get("requires_clarification"):
                        return MatchResult(
                            MatchMode.CLARIFICATION, 0.995, entry.get("category"),
                            entry.get("clarification_question"), hard_id,
                            [self._clean_entry_for_llm(entry)], entry=entry,
                            second_score=0.0, margin=0.995, fuzzy_score=1.0,
                            intent_fit_score=fit, intent_fit_reason=fit_reason,
                            family_fit_score=family_fit, query_quality_score=quality.score,
                            subquestion=question, decision_reason="hard_rule",
                        )
                    return MatchResult(
                        MatchMode.TEMPLATE, 0.995, entry.get("category"),
                        (entry.get("template_answer") or ""),
                        hard_id, [self._clean_entry_for_llm(entry)], entry=entry,
                        second_score=0.0, margin=0.995, fuzzy_score=1.0,
                        intent_fit_score=fit, intent_fit_reason=fit_reason,
                        family_fit_score=family_fit,
                        query_quality_score=quality.score,
                        subquestion=question, decision_reason="hard_rule",
                    )
        # 2. Obvious non-query/noise is rejected before expensive retrieval.
        # Short but semantically meaningful messages are not rejected here if
        # they have a strong fuzzy match.
        if quality.meta_noise:
            return MatchResult(
                MatchMode.NO_MATCH, 0.0, None, None, None, [],
                subquestion=question,
                decision_reason=f"noise:{quality.reason}",
                query_quality_score=quality.score,
            )

        fuzzy = self._fuzzy_candidates(question)
        if fuzzy:
            top_f = fuzzy[0]
            second_f = fuzzy[1]["score"] if len(fuzzy) > 1 else 0.0
            fuzzy_gap = top_f["score"] - second_f
            if (
                top_f["score"] >= 98.0
                and fuzzy_gap >= 8.0
                and not top_f["negative_hits"]
            ):
                entry = top_f["entry"]
                fit, fit_reason = self._intent_fit(
                    question, entry, fuzzy_phrase=top_f["score"] / 100.0,
                )
                family_fit = self._family_fit(question, entry)
                mode = MatchMode.CLARIFICATION if entry.get("requires_clarification") else MatchMode.TEMPLATE
                answer = (
                    entry.get("clarification_question")
                    if mode == MatchMode.CLARIFICATION
                    else (entry.get("template_answer") or "")
                )
                eligible, eligibility_reason = self._template_eligibility(
                    question, entry, intent_fit=fit, family_fit=family_fit,
                    fuzzy_score=top_f["score"] / 100.0,
                    hard_rule=False,
                )
                if fit >= 0.55 and family_fit >= 0.25 and answer and eligible:
                    return MatchResult(
                        mode, top_f["score"] / 100.0, entry.get("category"), answer,
                        entry.get("id"), [self._clean_entry_for_llm(entry)], entry=entry,
                        second_score=second_f / 100.0, margin=fuzzy_gap / 100.0,
                        fuzzy_score=top_f["score"] / 100.0,
                        intent_fit_score=fit, intent_fit_reason=fit_reason,
                        family_fit_score=family_fit,
                        query_quality_score=quality.score,
                        subquestion=question, decision_reason="near_duplicate_phrase",
                    )

        # 3. Hybrid retrieval + evidence-first ranking.
        candidates = self._retrieve_candidates(question)
        ranked = self._rank_candidates(question, candidates)
        if not ranked:
            return MatchResult(
                MatchMode.NO_MATCH, 0.0, None, None, None, [],
                subquestion=question, decision_reason="no_candidates",
                query_quality_score=quality.score,
            )

        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        top_score = float(top.get("final_score", 0.0))
        second_score = float(second.get("final_score", 0.0)) if second else 0.0
        margin = max(0.0, top_score - second_score)
        relevant_entries = self._build_relevant_entries(ranked)
        entry = top["entry"]
        intent_fit = float(top.get("intent_fit_score", 0.0))
        family_fit = float(top.get("family_fit_score", 0.0))
        real_evidence = float(top.get("real_intent_evidence", 0.0))
        direct_support = bool(top.get("direct_support", False))
        lexical = float(top.get("lexical_score", 0.0))
        fuzzy_score = float(top.get("fuzzy_phrase_score", 0.0))
        embedding = float(top.get("embedding_score", 0.0))
        semantic = float(top.get("semantic_score", 0.0))
        contradiction = bool(
            top.get("negative_hits") or
            (top.get("constraint_evidence") or {}).get("forbidden_hits")
        )
        top_candidates = [self._choice_candidate_payload(item) for item in ranked[:3]]
        negative_reason = self._negative_decision_reason(question, top, ranked)
        second_id = str(second["entry"]["id"]) if second else None
        conflict_ids = {str(x) for x in entry.get("conflict_ids", [])}
        second_is_conflict = bool(
            second_id
            and (
                second_id in conflict_ids
                or str(entry.get("id")) in {str(x) for x in (second.get("entry", {}).get("conflict_ids", []) or [])}
            )
        )
        contrastive_margin = float(top.get("contrastive_margin", 1.0))

        logger.info(
            "KB v12 decision | kb_id=%s | final=%.4f | margin=%.4f | emb=%.4f | sem=%.4f | lex=%.4f | fuzzy=%.4f | fit=%.4f | family=%.4f | evidence=%.4f | quality=%.4f | reason=%s",
            entry["id"], top_score, margin, embedding, semantic, lexical,
            fuzzy_score, intent_fit, family_fit, real_evidence,
            quality.score, top.get("intent_fit_reason", ""),
        )

        logger.debug(
            "contrastive | top=%s competitor=%s margin=%.4f reason=%s",
            entry["id"], top.get("contrastive_competitor_id"), contrastive_margin,
            top.get("contrastive_reason", ""),
        )

        if contradiction:
            return MatchResult(
                MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
                second_score=second_score, margin=margin,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason="contradiction",
                top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        if negative_reason and negative_reason not in {"explicit_candidate_negative"}:
            return MatchResult(
                MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
                second_score=second_score, margin=margin,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason=negative_reason,
                top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        # 3.5 Open-set gate. OOS is only emitted after noise/contradiction checks
        # and before ambiguity/template/LLM routing, so an unsupported message cannot
        # inherit a nearby category merely because embeddings are high.
        oos_reason = self._oos_reason(question, ranked, quality)
        if oos_reason:
            return MatchResult(
                MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
                second_score=second_score, margin=margin,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason=oos_reason,
                top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        # 4. Known ambiguity is a first-class state. If the second candidate is
        # a declared conflict and both have meaningful evidence, ask the user
        # instead of guessing.
        second_real = float(second.get("real_intent_evidence", 0.0)) if second else 0.0
        second_fit = float(second.get("intent_fit_score", 0.0)) if second else 0.0
        both_supported = (
            second is not None
            and second_is_conflict
            and second_real >= 0.42
            and second_fit >= 0.50
            and real_evidence >= 0.42
            and top_score < 0.88
            and contrastive_margin < CONTRASTIVE_AUTO_MARGIN
        )
        if both_supported and margin < 0.08:
            choices = self._category_choice_candidates(ranked)
            if len(choices) >= 2:
                return MatchResult(
                    MatchMode.CATEGORY_CHOICE, top_score, entry.get("category"), None,
                    entry.get("id"), relevant_entries, entry=entry,
                    second_score=second_score, margin=margin,
                    semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                    evidence_score=float(top.get("evidence_score", 0.0)),
                    real_intent_evidence=real_evidence,
                    embedding_score=embedding, intent_fit_score=intent_fit,
                    intent_fit_reason=str(top.get("intent_fit_reason", "")),
                    family_fit_score=family_fit, query_quality_score=quality.score,
                    subquestion=question, decision_reason="ambiguous_conflict",
                    choice_candidates=choices, top_candidates=top_candidates,
                    domain_support_score=float(top.get("domain_support_score", 0.0)),
                    ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                    negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                    contrastive_margin=contrastive_margin,
                    contrastive_competitor_id=top.get("contrastive_competitor_id"),
                    contrastive_reason=str(top.get("contrastive_reason", "")),
                )

        # 5.1. Intent-first deterministic override.
        # Some support queries have a modest aggregate score because several
        # intents share vocabulary, while their intent contract is explicit.
        domain_support = float(top.get("domain_support_score", 0.0))
        ontology_alignment = float(top.get("ontology_alignment_score", 0.0))
        second_real = float(second.get("real_intent_evidence", 0.0)) if second else 0.0
        secondary_ambiguity = (
            second is not None
            and second_real >= SECONDARY_EVIDENCE_MIN
            and margin < SECONDARY_EVIDENCE_MARGIN
        )
        deterministic_intent_safe = (
            intent_fit >= 0.90
            and domain_support >= 0.30
            and ontology_alignment >= 0.62
            and real_evidence >= 0.30
            and not contradiction
            and not second_is_conflict
            and not secondary_ambiguity
            and (direct_support or lexical >= 0.55 or fuzzy_score >= 0.90)
        )
        deterministic_eligible, deterministic_reason = self._template_eligibility(
            question, entry, intent_fit=intent_fit, family_fit=family_fit,
            fuzzy_score=fuzzy_score, contrastive_margin=contrastive_margin,
            second_is_conflict=second_is_conflict,
        )
        if (
            deterministic_intent_safe
            and deterministic_eligible
            and entry.get("template_answer")
            and top_score >= AUTO_SCORE
            and real_evidence >= MIN_TEMPLATE_EVIDENCE
        ):
            return MatchResult(
                MatchMode.TEMPLATE, top_score, entry.get("category"),
                (entry.get("template_answer") or ""),
                entry.get("id"), relevant_entries, entry=entry,
                second_score=second_score, margin=margin, embedding_score=embedding,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason="deterministic_high_intent",
                top_candidates=top_candidates, domain_support_score=domain_support,
                ontology_alignment_score=ontology_alignment,
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        # 5. If the evidence points to a supported intent, allow automatic
        # routing even when the query itself is short/grammatically rough.
        # Quality therefore influences OOS/noise detection, not basic intent
        # classification.
        direct_semantic_support = (
            embedding >= 0.84
            and semantic >= 0.67
            and intent_fit >= 0.68
            and family_fit >= 0.45
            and margin >= 0.07
        )
        direct_lexical_support = (
            (direct_support or lexical >= 0.50 or fuzzy_score >= 0.92)
            and intent_fit >= 0.58
            and family_fit >= 0.30
            and real_evidence >= 0.38
            and margin >= (0.035 if second_is_conflict else 0.025)
        )
        safe_auto = (
            (not second_is_conflict or margin >= 0.075)
            and (not second_is_conflict or contrastive_margin >= CONTRASTIVE_AUTO_MARGIN)
            and not secondary_ambiguity
        ) and (direct_lexical_support or direct_semantic_support)

        safe_template_eligible, safe_template_reason = self._template_eligibility(
            question, entry, intent_fit=intent_fit, family_fit=family_fit,
            fuzzy_score=fuzzy_score, contrastive_margin=contrastive_margin,
            second_is_conflict=second_is_conflict,
        )

        if entry.get("requires_clarification") and safe_auto and safe_template_eligible:
            return MatchResult(
                MatchMode.CLARIFICATION, top_score, entry.get("category"),
                entry.get("clarification_question"), entry.get("id"), relevant_entries,
                entry=entry, second_score=second_score, margin=margin,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason="supported_intent_requires_clarification",
            )

        # TEMPLATE only when aggregate score is near-certain (ultra-niche).
        if (
            safe_auto
            and safe_template_eligible
            and entry.get("template_answer")
            and top_score >= AUTO_SCORE
            and margin >= AUTO_MARGIN * 0.5
            and intent_fit >= INTENT_FIT_TEMPLATE_MIN
            and real_evidence >= MIN_TEMPLATE_EVIDENCE
        ):
            reason = (
                "strong_semantic_intent_match"
                if direct_semantic_support and not direct_lexical_support
                else "strong_lexical_intent_match"
            )
            return MatchResult(
                MatchMode.TEMPLATE, top_score, entry.get("category"),
                (entry.get("template_answer") or ""),
                entry.get("id"), relevant_entries, entry=entry,
                second_score=second_score, margin=margin, embedding_score=embedding,
                intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason=reason,
                top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        # 6. Primary middle path: Top-3 whenever 2+ candidates survive soft filters.
        choices = self._category_choice_candidates(ranked)
        if len(choices) >= 2:
            return MatchResult(
                MatchMode.CATEGORY_CHOICE, top_score, entry.get("category"), None,
                entry.get("id"), relevant_entries, entry=entry,
                second_score=second_score, margin=margin,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason="supported_ambiguity",
                choice_candidates=choices, top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
            )

        # 6b. Single strong-but-not-template candidate: still try Top-3 from ranked.
        if top_score >= 0.50 and len(top_candidates) >= 2:
            soft_choices = []
            for c in top_candidates[:3]:
                soft_choices.append(c if isinstance(c, dict) else {"choice_label": str(c), "kb_id": str(c)})
            # Ensure choice_label present
            normalized = []
            for c in soft_choices:
                if not c:
                    continue
                label = c.get("choice_label") or c.get("category") or c.get("kb_id")
                if not label:
                    continue
                normalized.append({**c, "choice_label": label})
            if len(normalized) >= 2:
                return MatchResult(
                    MatchMode.CATEGORY_CHOICE, top_score, entry.get("category"), None,
                    entry.get("id"), relevant_entries, entry=entry,
                    second_score=second_score, margin=margin,
                    semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                    evidence_score=float(top.get("evidence_score", 0.0)),
                    real_intent_evidence=real_evidence,
                    embedding_score=embedding, intent_fit_score=intent_fit,
                    intent_fit_reason=str(top.get("intent_fit_reason", "")),
                    family_fit_score=family_fit, query_quality_score=quality.score,
                    subquestion=question, decision_reason="mid_confidence_top3",
                    choice_candidates=normalized, top_candidates=top_candidates,
                    domain_support_score=float(top.get("domain_support_score", 0.0)),
                    ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                    negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                )

        # 7. LLM mode is a last resort signal for orchestration (LLM disabled in live:
        # pipeline converts this to Top-3 or rephrase). Prefer not to reach here.
        llm_candidate_support = (
            top_score >= LLM_SCORE
            and intent_fit >= INTENT_FIT_LLM_MIN
            and family_fit >= 0.25
            and real_evidence >= 0.28
            and float(top.get("domain_support_score", 0.0)) >= 0.18
            and not contradiction
        )
        llm_contrast_safe = (
            not second_is_conflict
            or contrastive_margin >= CONTRASTIVE_LLM_MARGIN
        )

        if llm_candidate_support and llm_contrast_safe:
            return MatchResult(
                MatchMode.LLM, top_score, entry.get("category"), None,
                entry.get("id"), relevant_entries, entry=entry,
                second_score=second_score, margin=margin,
                semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
                evidence_score=float(top.get("evidence_score", 0.0)),
                real_intent_evidence=real_evidence,
                embedding_score=embedding, intent_fit_score=intent_fit,
                intent_fit_reason=str(top.get("intent_fit_reason", "")),
                family_fit_score=family_fit, query_quality_score=quality.score,
                subquestion=question, decision_reason="supported_intent_needs_llm",
                choice_candidates=choices if len(choices) >= 2 else None,
                top_candidates=top_candidates,
                domain_support_score=float(top.get("domain_support_score", 0.0)),
                ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
                negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
                contrastive_margin=contrastive_margin,
                contrastive_competitor_id=top.get("contrastive_competitor_id"),
                contrastive_reason=str(top.get("contrastive_reason", "")),
            )

        # 8. Abstain when evidence is weak. This is preferable to a wrong answer.
        # Mark the reason explicitly so orchestration can distinguish OOS from
        # insufficient context from low-confidence known intent.
        if quality.score < 0.14 and not direct_support:
            reason = "insufficient_context"
        elif top_score < 0.50 and not direct_support:
            reason = "low_confidence"
        elif family_fit < 0.25 and not direct_support:
            reason = "oos_or_wrong_family"
        else:
            reason = "unsupported_or_ambiguous"

        return MatchResult(
            MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
            second_score=second_score, margin=margin,
            semantic_score=semantic, lexical_score=lexical, fuzzy_score=fuzzy_score,
            evidence_score=float(top.get("evidence_score", 0.0)),
            real_intent_evidence=real_evidence,
            embedding_score=embedding, intent_fit_score=intent_fit,
            intent_fit_reason=str(top.get("intent_fit_reason", "")),
            family_fit_score=family_fit, query_quality_score=quality.score,
            subquestion=question, decision_reason=reason,
            choice_candidates=None,
            top_candidates=top_candidates,
            domain_support_score=float(top.get("domain_support_score", 0.0)),
            ontology_alignment_score=float(top.get("ontology_alignment_score", 0.0)),
            negative_contrast_score=float(top.get("negative_contrast_score", 0.0)),
            contrastive_margin=contrastive_margin,
            contrastive_competitor_id=top.get("contrastive_competitor_id"),
            contrastive_reason=str(top.get("contrastive_reason", "")),
        )

    def validate_llm_selection(self, question: str, kb_id: str) -> tuple[bool, str, dict[str, float]]:
        """Re-check an LLM-selected intent with deterministic business evidence.

        The model can choose among candidates, but it cannot bypass hard KB
        contradictions. This protects against LLM preference drift in close
        semantic neighborhoods.
        """
        entry = self.entries_by_id.get(str(kb_id))
        if not entry:
            return False, "unknown_kb_id", {}

        negative_hits = self._negative_hits(question, entry)
        constraints = self._constraint_evidence(question, entry)
        fuzzy = self._best_fuzzy_phrase_score(question, entry)
        intent_fit, intent_reason = self._intent_fit(question, entry, fuzzy_phrase=fuzzy)
        family_fit = self._family_fit(question, entry)
        domain_support = self._domain_support(question, entry)
        alignment, alignment_reason = self._ontology_alignment(question, entry)
        positive_hits = self._lexical_phrase_hits(question, entry)
        signal_pos, signal_neg = self._intent_signal_hits(question, entry)

        if negative_hits or constraints.get("forbidden_hits") or signal_neg:
            return False, "llm_selection_contradicted", {
                "intent_fit": intent_fit,
                "family_fit": family_fit,
                "domain_support": domain_support,
                "ontology_alignment": alignment,
                "fuzzy": fuzzy,
            }
        if alignment <= ONTOLOGY_HARD_MISMATCH and self._query_ontology(question)["object"]:
            return False, "llm_selection_object_mismatch", {
                "intent_fit": intent_fit,
                "family_fit": family_fit,
                "domain_support": domain_support,
                "ontology_alignment": alignment,
                "fuzzy": fuzzy,
            }

        contract = self._intent_contract_evidence(question, entry)
        if contract.status in {"insufficient", "contradicted"}:
            return False, f"llm_selection_contract:{contract.reason}", {
                "intent_fit": intent_fit,
                "family_fit": family_fit,
                "domain_support": domain_support,
                "ontology_alignment": alignment,
                "fuzzy": fuzzy,
            }

        supported = bool(
            positive_hits
            or signal_pos
            or fuzzy >= 0.82
        )
        safe = (
            intent_fit >= 0.48
            and family_fit >= 0.25
            and domain_support >= 0.18
            and alignment >= 0.35
            and supported
        )
        reason = "llm_selection_valid" if safe else (
            f"llm_selection_weak:{intent_reason};{alignment_reason}"
        )
        return safe, reason, {
            "intent_fit": intent_fit,
            "family_fit": family_fit,
            "domain_support": domain_support,
            "ontology_alignment": alignment,
            "fuzzy": fuzzy,
        }

    # --------------------------------------------------------
    # FUZZY / DEBUG
    # --------------------------------------------------------

    def _fuzzy_candidates(self, question: str) -> list[dict]:
        q = normalize_light(question)
        result = []
        for entry in self.entries:
            best_score = 0.0
            best_phrase = ""
            for phrase in entry.get("positive_examples", []):
                p = normalize_light(phrase)
                if q == p:
                    score = 100.0
                else:
                    score = float(max(fuzz.ratio(q, p), fuzz.WRatio(q, p)))
                if score > best_score:
                    best_score = score
                    best_phrase = phrase
            result.append({
                "entry": entry,
                "score": best_score,
                "phrase": best_phrase,
                "negative_hits": self._negative_hits(question, entry),
            })
        return sorted(result, key=lambda x: x["score"], reverse=True)

    def get_relevant_entries(self, question: str, top_n: int = FINAL_CANDIDATES) -> list[dict]:
        candidates = self._retrieve_candidates(question, max(RETRIEVAL_TOP_K, top_n))
        ranked = self._rank_candidates(question, candidates)
        return self._build_relevant_entries(ranked[:top_n])

    def debug_match(self, question: str) -> dict:
        fuzzy = self._fuzzy_candidates(question)
        candidates = self._retrieve_candidates(question)
        ranked = self._rank_candidates(question, candidates)
        decision = self.match(question)
        return {
            "question": question,
            "decision": {
                "mode": decision.mode.value,
                "kb_id": decision.kb_id,
                "category": decision.category,
                "score": decision.score,
                "second_score": decision.second_score,
                "margin": decision.margin,
                "embedding_score": decision.embedding_score,
                "intent_fit_score": decision.intent_fit_score,
                "family_fit_score": decision.family_fit_score,
                "query_quality_score": decision.query_quality_score,
                "query_quality_reason": getattr(decision, "decision_reason", ""),
                "intent_fit_reason": decision.intent_fit_reason,
                "decision_reason": decision.decision_reason,
            },
            "hard_rule": self._hard_rule_match(question),
            "fuzzy": [
                {"kb_id": x["entry"]["id"], "score": x["score"], "phrase": x["phrase"], "negative_hits": x["negative_hits"]}
                for x in fuzzy[:5]
            ],
            "candidates": [
                {
                    "kb_id": x["entry"]["id"],
                    "family": x["entry"].get("family"),
                    "embedding_score": x["embedding_score"],
                    "semantic_top3": x["semantic_top3"],
                    "semantic_prototype": x["semantic_prototype"],
                    "bm25_score": x["bm25_score"],
                    "rrf_score": x["rrf_score"],
                    "rerank_score": x["rerank_score"],
                    "semantic_score": x["semantic_score"],
                    "rrf_normalized": x.get("rrf_normalized", 0.0),
                    "fuzzy_phrase_score": x.get("fuzzy_phrase_score", 0.0),
                    "required_evidence": x.get("required_evidence", {}),
                    "final_score": x["final_score"],
                    "positive_hits": x["positive_hits"],
                    "negative_hits": x["negative_hits"],
                    "best_phrase": x["best_phrase"],
                    "intent_fit_score": x.get("intent_fit_score", 0.0),
                    "family_fit_score": x.get("family_fit_score", 0.0),
                    "query_quality_score": x.get("query_quality_score", 0.0),
                    "query_quality_reason": x.get("query_quality_reason", ""),
                    "lexical_score": x.get("lexical_score", 0.0),
                    "evidence_score": x.get("evidence_score", 0.0),
                    "intent_fit_reason": x.get("intent_fit_reason", ""),
                    "ranking_score": x.get("ranking_score", x["final_score"]),
                    "choice_label": x["entry"].get("choice_label"),
                    "real_intent_evidence": x.get("real_intent_evidence", 0.0),
                    "direct_support": x.get("direct_support", False),
                    "contrastive_margin": x.get("contrastive_margin", 0.0),
                    "contrastive_competitor_id": x.get("contrastive_competitor_id"),
                    "contrastive_reason": x.get("contrastive_reason", ""),
                }
                for x in ranked
            ],
        }

    def _validate_conjunction_split(self, parts: list[str]) -> bool:
        """Accept an `и` split only when fragments resolve to distinct intents."""
        if len(parts) != 2:
            return False

        left = self.match(parts[0])
        right = self.match(parts[1])

        if left.mode == MatchMode.OPERATOR or right.mode == MatchMode.OPERATOR:
            return False
        if left.kb_id is None or right.kb_id is None:
            return False
        if left.kb_id == right.kb_id:
            return False

        left_confident = (
            left.mode in (
                MatchMode.TEMPLATE,
                MatchMode.CLARIFICATION,
                MatchMode.CATEGORY_CHOICE,
            )
            or (left.mode == MatchMode.LLM and left.score >= LLM_SCORE)
        )
        right_confident = (
            right.mode in (
                MatchMode.TEMPLATE,
                MatchMode.CLARIFICATION,
                MatchMode.CATEGORY_CHOICE,
            )
            or (right.mode == MatchMode.LLM and right.score >= LLM_SCORE)
        )
        if not (left_confident and right_confident):
            return False

        return True

    def match_many(self, question: str) -> list[MatchResult]:
        """Classify independent intents without blindly splitting on conjunctions."""
        question = (question or "").strip()
        if not question:
            return [self.match(question)]

        structural = split_user_questions(question)
        if len(structural) > 1:
            return [self.match(part) for part in structural[:MAX_SUBQUESTIONS]]

        conjunction_candidates = _split_conjunctive_question(question)
        if len(conjunction_candidates) == 2 and self._validate_conjunction_split(conjunction_candidates):
            return [self.match(part) for part in conjunction_candidates]

        return [self.match(question)]
