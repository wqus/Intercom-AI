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
FINAL_CANDIDATES = 6
LLM_TOP_K = 3

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
GENERIC_RERANK_WEIGHT = 0.10

# Decision thresholds are deliberately based on the composite ranking signal.
# These are routing thresholds, NOT probabilities. The important companion
# signal is the top-vs-second margin and the presence of lexical evidence.
AUTO_SCORE = 0.82
AUTO_MARGIN = 0.075
LLM_SCORE = 0.56
LLM_MARGIN = 0.020
HIGH_SCORE_LOW_MARGIN = 0.88
LEXICAL_EVIDENCE_MIN = 0.18
SEMANTIC_GAP_FOR_LLM = 0.020

# Additional deterministic evidence. These are ranking features, not probabilities.
FUZZY_PHRASE_WEIGHT = 0.10
REQUIRED_EVIDENCE_WEIGHT = 0.06
CONFLICT_COMPETITOR_MARGIN = 0.12

# Intent-fit layer: semantic similarity alone is not enough for template routing.
# Strong question-type mismatches (e.g. AGE vs PROCEDURE, DURATION vs PRICE)
# reduce fit and can block an unsafe automatic template.
INTENT_FIT_TEMPLATE_MIN = 0.55
INTENT_FIT_LLM_MIN = 0.28
INTENT_FIT_HARD_MISMATCH = 0.18

# Hard rejection for unusable input.
MIN_MEANINGFUL_CHARS = 3
MIN_ALNUM_RATIO = 0.45
MAX_REPEAT_CHAR_RATIO = 0.72

MAX_SUBQUESTIONS = 5

OPERATOR_HINT = (
    "\n\n---\n"
    "Если этот ответ не помог решить ваш вопрос, просто напишите «оператор». "
    "Мы повторно рассмотрим ваше обращение и ответим вам в этом чате."
)

# Words that MUST preserve meaning in support intents.
STOP_WORDS = {
    "могу", "как", "что", "это", "мне", "у", "меня", "мой", "моя", "мое", "моё",
    "можно", "нужно", "надо", "есть", "был", "была", "буду", "вы", "я", "он", "она",
    "пожалуйста", "помогите", "скажите", "подскажите", "ли", "а", "и", "же", "бы", "быть",
    # Deliberately NOT removing: не, нет, нельзя, уже, только, без, еще/ещё.
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
    embedding_score: float = 0.0
    intent_fit_score: float = 0.0
    intent_fit_reason: str = ""
    subquestion: str | None = None
    decision_reason: str = ""


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


def contains_operator_request(text: str) -> bool:
    q = normalize_light(text)
    return bool(re.search(r"\bоператор\b|соедините\s+с\s+оператором|переключите\s+на\s+оператора|хочу\s+оператора|позовите\s+оператора|нужен\s+оператор", q))


def _quality_gate(text: str) -> tuple[bool, str]:
    q = normalize_light(text)
    if len(q) < MIN_MEANINGFUL_CHARS:
        return False, "too_short"
    alnum = sum(ch.isalnum() for ch in q)
    if alnum / max(1, len(q.replace(" ", ""))) < MIN_ALNUM_RATIO:
        return False, "low_alnum_ratio"
    compact = q.replace(" ", "")
    if compact:
        repeats = sum(1 for a, b in zip(compact, compact[1:]) if a == b)
        if repeats / max(1, len(compact) - 1) > MAX_REPEAT_CHAR_RATIO:
            return False, "repeated_characters"
    tokens = tokenize(q)
    if not tokens:
        return False, "no_meaningful_tokens"
    # A long string consisting almost entirely of one invented token is likely garbage.
    if len(tokens) <= 3 and all(len(t) >= 6 for t in tokens) and not any(c.isdigit() for c in q):
        # Leave normal long Russian words alone; reject only highly consonant/noisy strings.
        vowels = sum(c in "аеёиоуыэюя" for c in compact)
        letters = sum(c.isalpha() for c in compact)
        if letters and vowels / letters < 0.18:
            return False, "nonsense_like"
    return True, "ok"


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
    meaningful = [x for x in result if len(tokenize(x)) >= 2]
    return meaningful[:MAX_SUBQUESTIONS] if len(meaningful) > 1 else [text]


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
        r"\s+и\s+",
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
    """
    structural = _split_structural_questions(text)
    if len(structural) > 1:
        return structural[:max_questions]
    return [text.strip()]


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
      5. Abstention: TEMPLATE / LLM / CLARIFICATION / UNKNOWN.

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
        self.default_answer = "Добрый день!\nУточните, пожалуйста, подробнее ваш вопрос."
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
            "KB v6 loaded | categories=%d | phrase_records=%d | reranker=%s",
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
                self.default_answer = str(raw.get("template_answer") or self.default_answer)
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
            entry["constraints"] = entry.get("constraints") if isinstance(entry.get("constraints"), dict) else {}

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

        self._validate_entries()

    def _validate_entries(self) -> None:
        """Fail fast on KB structure errors that would make ranking unreliable."""
        known_ids = set(self.entries_by_id)
        for entry in self.entries:
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
        for entry in self.entries:
            idxs: list[int] = []
            # We intentionally index only positive examples. Negative examples
            # are used as contradiction evidence, never as retrieval documents.
            for phrase in entry["positive_examples"]:
                idx = len(passages)
                self.phrase_records.append({
                    "entry_id": entry["id"],
                    "phrase": phrase,
                    "index": idx,
                })
                self.category_phrase_indices.setdefault(entry["id"], []).append(idx)
                passages.append(f"passage: {phrase}")
                docs.append(tokenize(phrase))
                idxs.append(idx)

            # A compact category prototype is also indexed semantically.
            prototype = self._category_prototype(entry)
            emb = self.model.encode(
                [f"passage: {prototype}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            self.category_embeddings[entry["id"]] = np.asarray(emb, dtype=np.float32)

        if passages:
            embeddings = self.model.encode(
                passages,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )
            self.phrase_embeddings_matrix = np.asarray(embeddings, dtype=np.float32)
        else:
            self.phrase_embeddings_matrix = np.empty((0, 0), dtype=np.float32)

        # One lexical document per intent: all positive examples concatenated.
        self.intent_docs: list[list[str]] = [
            tokenize(" ".join(entry["positive_examples"])) for entry in self.entries
        ]
        self.bm25 = BM25Index(self.intent_docs)
        self.intent_index = {entry["id"]: i for i, entry in enumerate(self.entries)}

    @staticmethod
    def _category_prototype(entry: dict) -> str:
        category = str(entry.get("category") or "")
        family = str(entry.get("family") or "")
        definition = str(entry.get("definition") or entry.get("scope") or entry.get("description") or "")
        positives = ". ".join(str(x) for x in entry.get("positive_examples", [])[:10])
        return (
            f"Семейство: {family}. Категория: {category}. "
            f"Определение: {definition}. Положительные примеры: {positives}."
        )

    # --------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------

    def get_by_id(self, kb_id: str) -> dict | None:
        return self.entries_by_id.get(str(kb_id))

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

        def has_phrase(*phrases: str) -> bool:
            return any(normalize_light(p) in q for p in phrases)

        def has_school() -> bool:
            return any(token.startswith("школ") for token in tokens)

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
        # ----------------------------------------------------
        if has_any("стать") and has_any("настав", "ментор") and not has_phrase("мой наставник"):
            return "kb_023"
        if has_any("программ") and has_any("наставнич") and not has_any("мой настав", "ментор"):
            return "kb_034"
        if has_any("заявк") and has_any("настав", "ментор") and has_any("повтор", "снов", "отказ", "отклон"):
            return "kb_025"
        if has_any("настав", "ментор") and not has_phrase("стать наставником", "стать ментором"):
            return "kb_009"

        # ----------------------------------------------------
        # Education entry / school registration / registration failure.
        # ----------------------------------------------------
        registration_failure = (
            has_any("запис", "вступ", "попаст") and
            (has_any("не", "нет") or has_phrase("ошибка при записи", "нет кнопки записи", "не пускает в школу"))
        )
        school_registration = has_school() and has_any("запис", "вступ", "попаст", "выбрат")
        general_training = has_any("обуч", "учит", "учеб")
        if registration_failure and school_registration:
            return "kb_002"
        if school_registration and not registration_failure:
            return "kb_033"

        # Explicit 'how to start/get into training' must not be stolen by kb_033.
        if general_training and has_any("попаст", "начат", "старт", "попад", "участв") and not school_registration:
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
            result[entry["id"]] = {
                "max": max_score,
                "top3": top3,
                "prototype": proto,
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
            sem = semantic.get(entry["id"], {"max": 0.0, "top3": 0.0, "prototype": 0.0})
            phrase_idx = self._best_phrase_index(entry["id"], question)
            result.append({
                "entry": entry,
                "embedding_score": float(sem["max"]),
                "semantic_top3": float(sem["top3"]),
                "semantic_prototype": float(sem["prototype"]),
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

        patterns: dict[str, tuple[str, ...]] = {
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

    def _rank_candidates(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        pairs = []
        for item in candidates:
            entry = item["entry"]
            pairs.append((question, self._reranker_text(entry, item["best_phrase"])))

        generic_scores = [0.0] * len(candidates)
        if self.reranker is not None:
            raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
            generic_scores = [self._sigmoid(float(np.asarray(x).reshape(-1)[0])) for x in raw_scores]

        ranked = []
        for item, generic in zip(candidates, generic_scores):
            entry = item["entry"]
            positives = self._lexical_phrase_hits(question, entry)
            negatives = self._negative_hits(question, entry)
            constraints = self._constraint_evidence(question, entry)
            required = self._required_evidence(question, entry)
            fuzzy_phrase = self._best_fuzzy_phrase_score(question, entry)
            intent_fit, intent_fit_reason = self._intent_fit(
                question, entry, fuzzy_phrase=fuzzy_phrase
            )

            semantic = (
                SEMANTIC_MAX_WEIGHT * item["embedding_score"]
                + SEMANTIC_TOP3_WEIGHT * item["semantic_top3"]
                + SEMANTIC_CATEGORY_WEIGHT * item["semantic_prototype"]
            )
            # RRF is a rank-fusion signal. Normalize it against its theoretical
            # best-case value when the candidate is first in both rank lists.
            rrf_max = 2.0 / (RRF_K + 1.0)
            rrf_norm = min(1.0, item["rrf_score"] / max(rrf_max, 1e-9))

            base = 0.68 * semantic + 0.20 * rrf_norm + 0.12 * fuzzy_phrase
            lexical_bonus = POSITIVE_LEXICAL_BONUS if positives else 0.0
            required_bonus = REQUIRED_EVIDENCE_WEIGHT * required["coverage"]
            negative_penalty = NEGATIVE_LEXICAL_PENALTY if negatives else 0.0
            conflict_penalty = CONFLICT_PENALTY if constraints["forbidden_hits"] else 0.0
            group_bonus = REQUIRED_GROUP_BONUS if constraints["failed_any_groups"] == 0 and constraints["constraint_groups"] else 0.0

            final = (
                base
                + lexical_bonus
                + required_bonus
                + group_bonus
                - negative_penalty
                - conflict_penalty
                + GENERIC_RERANK_WEIGHT * generic
            )
            # Keep the original final_score for continuity, but use a small
            # intent-fit adjustment for candidate ordering. A semantically similar
            # category with the wrong question type should not outrank a candidate
            # whose intent contract actually matches the question.
            ranking_score = final + 0.16 * (intent_fit - 0.5)

            item = dict(item)
            item.update({
                "rerank_score": generic,
                "semantic_score": float(semantic),
                "rrf_normalized": float(rrf_norm),
                "fuzzy_phrase_score": float(fuzzy_phrase),
                "required_evidence": required,
                "intent_fit_score": float(intent_fit),
                "intent_fit_reason": intent_fit_reason,
                "ranking_score": float(ranking_score),
                "final_score": float(final),
                "positive_hits": positives,
                "negative_hits": negatives,
                "constraint_evidence": constraints,
            })
            ranked.append(item)

        ranked.sort(key=lambda x: x.get("ranking_score", x["final_score"]), reverse=True)
        return ranked[:FINAL_CANDIDATES]

    @staticmethod
    def _reranker_text(entry: dict, best_phrase: str) -> str:
        category = str(entry.get("category") or "")
        family = str(entry.get("family") or "")
        intent_type = str(entry.get("intent_type") or "")
        facets = ", ".join(str(x) for x in entry.get("intent_facets", [])[:8])
        definition = str(entry.get("definition") or entry.get("scope") or entry.get("description") or "")
        positives = "; ".join(str(x) for x in entry.get("positive_examples", [])[:8])
        negatives = "; ".join(str(x) for x in entry.get("negative_phrases", [])[:5])
        signal_pos = "; ".join(str(x) for x in (entry.get("intent_signals") or {}).get("positive", [])[:8])
        signal_neg = "; ".join(str(x) for x in (entry.get("intent_signals") or {}).get("negative", [])[:8])
        return (
            f"Семейство: {family}. Тип интента: {intent_type}. Фасеты: {facets}. "
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
            "clarification_question", "template_answer", "description", "hard_negative_examples", "intent_type", "intent_facets", "intent_signals",
        }
        return {k: v for k, v in entry.items() if k in allowed}

    def _build_relevant_entries(self, ranked: list[dict]) -> list[dict]:
        result = []
        for item in ranked[:LLM_TOP_K]:
            clean = self._clean_entry_for_llm(item["entry"])
            clean.update({
                "_embedding_score": float(item["embedding_score"]),
                "_semantic_top3": float(item["semantic_top3"]),
                "_semantic_prototype": float(item["semantic_prototype"]),
                "_bm25_score": float(item["bm25_score"]),
                "_rrf_score": float(item["rrf_score"]),
                "_rerank_score": float(item["rerank_score"]),
                "_semantic_score": float(item["semantic_score"]),
                "_rrf_normalized": float(item.get("rrf_normalized", 0.0)),
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
                "_intent_type": item["entry"].get("intent_type"),
                "_intent_facets": item["entry"].get("intent_facets", []),

            })
            result.append(clean)
        return result

    # --------------------------------------------------------
    # MAIN MATCH
    # --------------------------------------------------------

    def match(self, question: str) -> MatchResult:
        question = (question or "").strip()
        if not question:
            return MatchResult(MatchMode.CLARIFICATION, 0.0, None, self.default_answer, None, [], subquestion=question)

        if contains_operator_request(question):
            return MatchResult(MatchMode.OPERATOR, 1.0, None, None, None, [], subquestion=question)

        ok, reason = _quality_gate(question)
        if not ok:
            logger.info("KB v6 input rejected as unknown | reason=%s | question=%r", reason, question)
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        if not self.entries:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        # 1) High-confidence deterministic business rule.
        # Rules are evaluated before fuzzy/semantic retrieval so explicit intent
        # boundaries (e.g. leave vs transfer vs restore) cannot be stolen by a
        # semantically similar category.
        hard_id = self._hard_rule_match(question)
        if hard_id:
            entry = self.entries_by_id.get(hard_id)
            if entry and not self._negative_hits(question, entry):
                logger.info("KB v6 hard rule | kb_id=%s | question=%r", hard_id, question)
                if entry.get("requires_clarification"):
                    return MatchResult(MatchMode.CLARIFICATION, 0.99, entry.get("category"), entry.get("clarification_question"), hard_id, [self._clean_entry_for_llm(entry)], entry=entry, margin=0.99, subquestion=question)
                return MatchResult(MatchMode.TEMPLATE, 0.99, entry.get("category"), (entry.get("template_answer") or "") + OPERATOR_HINT, hard_id, [self._clean_entry_for_llm(entry)], entry=entry, margin=0.99, subquestion=question, decision_reason="hard_rule")

        # 2) Safe fuzzy shortcut. Fuzzy may only auto-answer when the phrase
        # match is extremely strong and the result is clearly separated.
        fuzzy = self._fuzzy_candidates(question)
        if fuzzy:
            top_f = fuzzy[0]
            second_f = fuzzy[1]["score"] if len(fuzzy) > 1 else 0.0
            if top_f["score"] >= 96.0 and top_f["score"] - second_f >= 8.0 and not top_f["negative_hits"]:
                entry = top_f["entry"]
                if entry.get("requires_clarification"):
                    return MatchResult(MatchMode.CLARIFICATION, top_f["score"] / 100.0, entry.get("category"), entry.get("clarification_question"), entry.get("id"), [self._clean_entry_for_llm(entry)], entry=entry, second_score=second_f / 100.0, margin=(top_f["score"] - second_f) / 100.0, fuzzy_score=top_f["score"] / 100.0, subquestion=question, decision_reason="strong_fuzzy_match")
                return MatchResult(MatchMode.TEMPLATE, top_f["score"] / 100.0, entry.get("category"), (entry.get("template_answer") or "") + OPERATOR_HINT, entry.get("id"), [self._clean_entry_for_llm(entry)], entry=entry, second_score=second_f / 100.0, margin=(top_f["score"] - second_f) / 100.0, fuzzy_score=top_f["score"] / 100.0, subquestion=question, decision_reason="strong_fuzzy_match")

        # 3) Hybrid retrieval.
        candidates = self._retrieve_candidates(question)
        ranked = self._rank_candidates(question, candidates)
        if not ranked:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        top_score = float(top["final_score"])
        second_score = float(second["final_score"]) if second else 0.0
        margin = top_score - second_score if second else top_score
        relevant_entries = self._build_relevant_entries(ranked)

        logger.info(
            "KB v5 decision | kb_id=%s | final=%.4f | margin=%.4f | emb=%.4f | sem=%.4f | bm25=%.4f | rrf=%.4f | rerank=%.4f | fit=%.4f | fit_reason=%s | pos=%s | neg=%s",
            top["entry"]["id"], top_score, margin, top["embedding_score"], top["semantic_score"], top["bm25_score"], top["rrf_score"], top["rerank_score"], top.get("intent_fit_score", 0.0), top.get("intent_fit_reason", ""), top["positive_hits"], top["negative_hits"],
        )

        entry = top["entry"]
        has_lexical = (
            bool(top["positive_hits"])
            or float(top["bm25_score"]) >= LEXICAL_EVIDENCE_MIN
            or float(top.get("fuzzy_phrase_score", 0.0)) >= 0.88
        )
        semantic_gap = (
            float(top["semantic_score"]) - float(second["semantic_score"])
            if second else float(top["semantic_score"])
        )
        top_conflicts = set(str(x) for x in entry.get("conflict_ids", []))
        second_is_known_conflict = bool(second and str(second["entry"]["id"]) in top_conflicts)
        intent_fit = float(top.get("intent_fit_score", 0.0))
        intent_fit_reason = str(top.get("intent_fit_reason", ""))
        hard_intent_mismatch = intent_fit < INTENT_FIT_HARD_MISMATCH

        if entry.get("requires_clarification"):
            return MatchResult(
                MatchMode.CLARIFICATION, top_score, entry.get("category"),
                entry.get("clarification_question"), entry.get("id"),
                relevant_entries, entry=entry, second_score=second_score,
                margin=margin, embedding_score=float(top["embedding_score"]),
                subquestion=question, decision_reason="kb_requires_clarification",
            )

        # A strong question-type mismatch blocks automatic template routing.
        # If the candidate remains semantically plausible, the LLM may inspect the
        # candidate set; otherwise the query is unresolved.
        if hard_intent_mismatch:
            if top_score >= LLM_SCORE and (has_lexical or semantic_gap >= SEMANTIC_GAP_FOR_LLM):
                return MatchResult(
                    MatchMode.LLM, top_score, entry.get("category"), None, entry.get("id"),
                    relevant_entries, entry=entry, second_score=second_score, margin=margin,
                    embedding_score=float(top["embedding_score"]),
                    intent_fit_score=intent_fit, intent_fit_reason=intent_fit_reason,
                    subquestion=question, decision_reason="intent_fit_mismatch_requires_llm",
                )
            return MatchResult(
                MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
                entry=None, second_score=second_score, margin=margin,
                embedding_score=float(top["embedding_score"]),
                intent_fit_score=intent_fit, intent_fit_reason=intent_fit_reason,
                subquestion=question, decision_reason="intent_fit_mismatch",
            )

        # A contradiction is strong evidence against automatic template routing.
        # If the candidate is nevertheless plausible, let the constrained LLM inspect it.
        if top["negative_hits"] or top["constraint_evidence"]["forbidden_hits"]:
            if (
                top_score >= LLM_SCORE
                and (has_lexical or semantic_gap >= SEMANTIC_GAP_FOR_LLM)
            ):
                return MatchResult(
                    MatchMode.LLM, top_score, entry.get("category"), None, entry.get("id"),
                    relevant_entries, entry=entry, second_score=second_score, margin=margin,
                    embedding_score=float(top["embedding_score"]), subquestion=question,
                    decision_reason="contradiction_requires_llm",
                )
            return MatchResult(
                MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries,
                entry=None, second_score=second_score, margin=margin,
                embedding_score=float(top["embedding_score"]), subquestion=question,
                decision_reason="strong_contradiction",
            )

        # Very strong and well-separated candidate -> deterministic template.
        # A strong score with weak separation is intentionally NOT auto-accepted.
        if (
            top_score >= AUTO_SCORE
            and margin >= AUTO_MARGIN
            and intent_fit >= INTENT_FIT_TEMPLATE_MIN
            and (not second_is_known_conflict or margin >= CONFLICT_COMPETITOR_MARGIN)
        ):
            return MatchResult(
                MatchMode.TEMPLATE, top_score, entry.get("category"),
                (entry.get("template_answer") or "") + OPERATOR_HINT, entry.get("id"),
                relevant_entries, entry=entry, second_score=second_score, margin=margin,
                embedding_score=float(top["embedding_score"]), intent_fit_score=intent_fit, intent_fit_reason=intent_fit_reason,
                subquestion=question, decision_reason="high_score_high_margin",
            )

        # Strong candidate but close second: send to LLM instead of dropping to NO_MATCH.
        # This fixes cases such as final=1.00, margin=0.028 where the top result
        # is clearly relevant but two intents are close.
        llm_eligible = (
            top_score >= LLM_SCORE
            and intent_fit >= INTENT_FIT_LLM_MIN
            and (
                margin >= LLM_MARGIN
                or (top_score >= HIGH_SCORE_LOW_MARGIN and (has_lexical or semantic_gap >= SEMANTIC_GAP_FOR_LLM))
                or (second_is_known_conflict and top_score >= LLM_SCORE)
                or intent_fit < INTENT_FIT_TEMPLATE_MIN
            )
        )
        if llm_eligible:
            return MatchResult(
                MatchMode.LLM, top_score, entry.get("category"), None, entry.get("id"),
                relevant_entries, entry=entry, second_score=second_score, margin=margin,
                embedding_score=float(top["embedding_score"]), intent_fit_score=intent_fit, intent_fit_reason=intent_fit_reason,
                subquestion=question, decision_reason="plausible_but_ambiguous",
            )

        # Weak candidate: current product policy still escalates NO_MATCH.
        return MatchResult(
            MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries, entry=None,
            second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]),
            intent_fit_score=intent_fit, intent_fit_reason=intent_fit_reason,
            subquestion=question, decision_reason="weak_or_unresolved",
        )

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
                    "intent_fit_reason": x.get("intent_fit_reason", ""),
                    "ranking_score": x.get("ranking_score", x["final_score"]),
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
            left.mode in (MatchMode.TEMPLATE, MatchMode.CLARIFICATION)
            or (left.mode == MatchMode.LLM and left.score >= LLM_SCORE)
        )
        right_confident = (
            right.mode in (MatchMode.TEMPLATE, MatchMode.CLARIFICATION)
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
