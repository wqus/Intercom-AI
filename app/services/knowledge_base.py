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

# Final acceptance is based on relative evidence, not generic model probability.
AUTO_SCORE = 0.64
LLM_SCORE = 0.45
AUTO_MARGIN = 0.10
LLM_MARGIN = 0.035

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
    "это", "мне", "у", "меня", "мой", "моя", "мое", "моё",
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
    subquestion: str | None = None


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
            if item:
                result.append(item)
    meaningful = [x for x in result if len(tokenize(x)) >= 2]
    return meaningful[:MAX_SUBQUESTIONS] if len(meaningful) > 1 else [text]


def _split_conjunctive_question(text: str) -> list[str]:
    q = text.strip()
    for pattern in (r"\s+а\s+также\s+", r"\s+и еще\s+", r"\s+и\s+"):
        parts = [p.strip(" ,") for p in re.split(pattern, q, maxsplit=1, flags=re.IGNORECASE)]
        if len(parts) == 2 and all(len(tokenize(p)) >= 3 for p in parts):
            return parts
    return [q]


def split_user_questions(text: str, max_questions: int = MAX_SUBQUESTIONS) -> list[str]:
    structural = _split_structural_questions(text)
    if len(structural) > 1:
        return structural[:max_questions]
    return _split_conjunctive_question(text)[:max_questions]


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
            "KB v2 loaded | categories=%d | phrase_records=%d | reranker=%s",
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
                continue

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
        not_use = ". ".join(str(x) for x in entry.get("when_not_to_use", [])[:5])
        return (
            f"Семейство: {family}. Категория: {category}. "
            f"Определение: {definition}. Положительные примеры: {positives}. "
            f"Не использовать: {not_use}."
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
        q = normalize_light(question)
        matches: list[str] = []
        for entry_id, positives, negatives in HARD_INTENT_RULES:
            if not any(normalize_light(p) in q for p in positives):
                continue
            if any(normalize_light(n) in q for n in negatives):
                continue
            matches.append(entry_id)

        # Special boundaries for education intents.
        failure = any(x in q for x in (
            "не могу запис", "не получается запис", "не дает запис", "нет кнопки записи",
            "ошибка при записи", "не пускает", "не работает запись",
        ))
        school_specific = any(x in q for x in ("в школу", "школу", "конкретную школу"))
        global_training = any(x in q for x in (
            "как попасть на обучение", "как начать обучение", "как начать учиться",
            "как стать участником обучения", "хочу попасть на обучение", "хочу учиться", "хочу начать учиться",
            "как попасть на поток", "как попасть на запуск", "с чего начать обучение",
        )) and not any(x in q for x in ("не могу", "не получается", "ошибка", "не работает"))

        if failure and school_specific:
            return "kb_002"
        if not failure and school_specific and any(x in q for x in ("как записаться", "как попасть", "где выбрать", "выбрать школу", "как вступить")):
            return "kb_033"
        # Robust phrase variants where users insert pronouns or context words.
        if ("не добавили" in q or "добавьте меня" in q or "не включили" in q) and any(x in q for x in ("чат", "групп", "телеграм", "тг")):
            return "kb_021"
        if any(x in q for x in ("восстанов", "вернуться", "обратно")) and any(x in q for x in ("отчис", "исключ", "ушел", "ушла", "вышел", "вышла")):
            return "kb_008"
        if not failure and global_training and not school_specific:
            return "kb_035"
        if ("попасть" in q or "начать" in q) and "обучение" in q and not school_specific and not failure:
            return "kb_035"

        # Do not return a rule merely because a very generic phrase is present.
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer the most specific (longest trigger) category.
            scored = []
            for entry_id in matches:
                triggers = next((p for i, p, _ in HARD_INTENT_RULES if i == entry_id), ())
                best = max((len(normalize_light(x)) for x in triggers if normalize_light(x) in q), default=0)
                scored.append((best, self.entries_by_id[entry_id].get("priority", 0), entry_id))
            scored.sort(reverse=True)
            return scored[0][2]
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

    def _negative_hits(self, question: str, entry: dict) -> list[str]:
        q = normalize_light(question)
        return [p for p in entry.get("negative_phrases", []) if isinstance(p, str) and normalize_light(p) in q]

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

            semantic = (
                SEMANTIC_MAX_WEIGHT * item["embedding_score"]
                + SEMANTIC_TOP3_WEIGHT * item["semantic_top3"]
                + SEMANTIC_CATEGORY_WEIGHT * item["semantic_prototype"]
            )
            # Combine semantic with rank fusion; do not interpret either as probability.
            base = 0.72 * semantic + 0.28 * min(1.0, item["rrf_score"] / (1.0 / (RRF_K + 1.0) * 2.0))
            lexical_bonus = POSITIVE_LEXICAL_BONUS if positives else 0.0
            negative_penalty = NEGATIVE_LEXICAL_PENALTY if negatives else 0.0
            conflict_penalty = CONFLICT_PENALTY if constraints["forbidden_hits"] else 0.0
            group_bonus = REQUIRED_GROUP_BONUS if constraints["failed_any_groups"] == 0 and constraints["constraint_groups"] else 0.0

            final = (
                base
                + lexical_bonus
                + group_bonus
                - negative_penalty
                - conflict_penalty
                + GENERIC_RERANK_WEIGHT * generic
            )

            item = dict(item)
            item.update({
                "rerank_score": generic,
                "semantic_score": float(semantic),
                "final_score": float(final),
                "positive_hits": positives,
                "negative_hits": negatives,
                "constraint_evidence": constraints,
            })
            ranked.append(item)

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked[:FINAL_CANDIDATES]

    @staticmethod
    def _reranker_text(entry: dict, best_phrase: str) -> str:
        category = str(entry.get("category") or "")
        family = str(entry.get("family") or "")
        definition = str(entry.get("definition") or entry.get("scope") or entry.get("description") or "")
        positives = "; ".join(str(x) for x in entry.get("positive_examples", [])[:8])
        negatives = "; ".join(str(x) for x in entry.get("negative_phrases", [])[:5])
        return (
            f"Семейство: {family}. Категория: {category}. "
            f"Определение: {definition}. Положительные примеры: {positives}. "
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
                "_final_score": float(item["final_score"]),
                "_best_match_phrase": item["best_phrase"],
                "_positive_hits": item["positive_hits"],
                "_negative_hits": item["negative_hits"],
                "_constraint_evidence": item["constraint_evidence"],
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
            logger.info("KB v2 input rejected as unknown | reason=%s | question=%r", reason, question)
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        if not self.entries:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        # 1) Safe exact/fuzzy only when the full query strongly matches and no contradiction exists.
        fuzzy = self._fuzzy_candidates(question)
        if fuzzy:
            top_f = fuzzy[0]
            second_f = fuzzy[1]["score"] if len(fuzzy) > 1 else 0.0
            if top_f["score"] >= 96.0 and top_f["score"] - second_f >= 8.0 and not top_f["negative_hits"]:
                entry = top_f["entry"]
                if entry.get("requires_clarification"):
                    return MatchResult(MatchMode.CLARIFICATION, top_f["score"] / 100.0, entry.get("category"), entry.get("clarification_question"), entry.get("id"), [self._clean_entry_for_llm(entry)], entry=entry, second_score=second_f / 100.0, margin=(top_f["score"] - second_f) / 100.0, fuzzy_score=top_f["score"] / 100.0, subquestion=question)
                return MatchResult(MatchMode.TEMPLATE, top_f["score"] / 100.0, entry.get("category"), (entry.get("template_answer") or "") + OPERATOR_HINT, entry.get("id"), [self._clean_entry_for_llm(entry)], entry=entry, second_score=second_f / 100.0, margin=(top_f["score"] - second_f) / 100.0, fuzzy_score=top_f["score"] / 100.0, subquestion=question)

        # 2) High-confidence deterministic rule.
        hard_id = self._hard_rule_match(question)
        if hard_id:
            entry = self.entries_by_id.get(hard_id)
            if entry and not self._negative_hits(question, entry):
                logger.info("KB v2 hard rule | kb_id=%s | question=%r", hard_id, question)
                if entry.get("requires_clarification"):
                    return MatchResult(MatchMode.CLARIFICATION, 0.99, entry.get("category"), entry.get("clarification_question"), hard_id, [self._clean_entry_for_llm(entry)], entry=entry, margin=0.99, subquestion=question)
                return MatchResult(MatchMode.TEMPLATE, 0.99, entry.get("category"), (entry.get("template_answer") or "") + OPERATOR_HINT, hard_id, [self._clean_entry_for_llm(entry)], entry=entry, margin=0.99, subquestion=question)

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
            "KB v2 decision | kb_id=%s | final=%.4f | margin=%.4f | emb=%.4f | sem=%.4f | bm25=%.4f | rrf=%.4f | rerank=%.4f | pos=%s | neg=%s",
            top["entry"]["id"], top_score, margin, top["embedding_score"], top["semantic_score"], top["bm25_score"], top["rrf_score"], top["rerank_score"], top["positive_hits"], top["negative_hits"],
        )

        entry = top["entry"]
        if entry.get("requires_clarification"):
            return MatchResult(MatchMode.CLARIFICATION, top_score, entry.get("category"), entry.get("clarification_question"), entry.get("id"), relevant_entries, entry=entry, second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]), subquestion=question)

        # If a direct lexical contradiction is present and the alternative is strong,
        # do not auto-accept the top semantic result.
        if top["negative_hits"]:
            return MatchResult(MatchMode.LLM if margin >= 0.02 else MatchMode.NO_MATCH, top_score, entry.get("category") if margin >= 0.02 else None, None, entry.get("id") if margin >= 0.02 else None, relevant_entries, entry=entry if margin >= 0.02 else None, second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]), subquestion=question)

        if top_score >= AUTO_SCORE and margin >= AUTO_MARGIN:
            return MatchResult(MatchMode.TEMPLATE, top_score, entry.get("category"), (entry.get("template_answer") or "") + OPERATOR_HINT, entry.get("id"), relevant_entries, entry=entry, second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]), subquestion=question)

        if top_score >= LLM_SCORE and margin >= LLM_MARGIN:
            return MatchResult(MatchMode.LLM, top_score, entry.get("category"), None, entry.get("id"), relevant_entries, entry=entry, second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]), subquestion=question)

        return MatchResult(MatchMode.NO_MATCH, top_score, None, None, None, relevant_entries, entry=None, second_score=second_score, margin=margin, embedding_score=float(top["embedding_score"]), subquestion=question)

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
                    "final_score": x["final_score"],
                    "positive_hits": x["positive_hits"],
                    "negative_hits": x["negative_hits"],
                    "best_phrase": x["best_phrase"],
                }
                for x in ranked
            ],
        }

    def match_many(self, question: str) -> list[MatchResult]:
        question = (question or "").strip()
        parts = split_user_questions(question)
        if len(parts) <= 1:
            return [self.match(question)]
        results = [self.match(part) for part in parts]
        # If splitting produced a low-confidence fragment, classify the original
        # message as one unit instead of hallucinating multiple intents.
        if any(r.mode == MatchMode.NO_MATCH for r in results):
            whole = self.match(question)
            if whole.mode != MatchMode.NO_MATCH:
                return [whole]
        return results[:MAX_SUBQUESTIONS]
