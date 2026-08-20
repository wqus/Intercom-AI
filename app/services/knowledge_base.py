from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger("knowledge_base")


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Retrieval is optimized for recall; CrossEncoder is optimized for precision.
RETRIEVAL_TOP_K = 12
RERANK_TOP_K = 5

# Fuzzy only short-circuits when the phrase is extremely close and there is
# enough separation from the next category.
FUZZY_EXACT_THRESHOLD = 97.0
FUZZY_MARGIN_THRESHOLD = 10.0

# E5 cosine similarity is not probability. It is used as a retrieval gate.
EMBEDDING_MIN_SCORE = 0.50
LLM_MIN_EMBEDDING_SCORE = 0.53

# CrossEncoder returns MS-MARCO logits. We convert them through sigmoid to a
# 0..1 relevance-like score. This is NOT a calibrated probability; thresholds
# should be tuned using your own labeled support questions.
RERANK_AUTO_THRESHOLD = 0.74
RERANK_LLM_THRESHOLD = 0.48
RERANK_MARGIN_THRESHOLD = 0.06

# Business evidence can move ranking, but never overwhelms semantic evidence.
REQUIRED_PENALTY = 0.15
NEGATIVE_PENALTY = 0.25
REQUIRED_BONUS = 0.05
PRIORITY_TIE_WEIGHT = 0.015

MAX_SUBQUESTIONS = 5

OPERATOR_HINT = (
    "\n\n---\n"
    "Если этот ответ не помог решить ваш вопрос, просто напишите «оператор». "
    "Мы повторно рассмотрим ваше обращение и ответим вам в этом чате."
)

STOP_WORDS = {
    "не", "могу", "как", "хочу", "что", "это", "мне", "у", "меня",
    "мой", "моя", "моё", "мое", "можно", "нужно", "надо", "есть", "был",
    "была", "буду", "вы", "я", "он", "она", "пожалуйста", "помогите",
    "скажите", "подскажите", "ли", "а", "и", "же", "бы", "быть",
}

# Narrow high-confidence business overrides for the most common conflict pairs.
ENTRY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kb_024", ("из всех школ", "из всех направлений", "отовсюду", "из всего обучения", "нигде не учиться")),
    ("kb_006", ("сменить школу", "перейти в другую школу", "перевестись в другую школу", "сменить направление")),
    ("kb_008", ("восстановиться в школе", "вернуться в школу", "вернуться после отчисления", "вернуться после ухода", "хочу обратно в школу", "восстановите меня")),
    ("kb_007", ("исключили из школы", "выгнали из школы", "отчислили из школы", "меня исключили", "меня отчислили")),
    ("kb_005", ("выйти из школы", "уйти из школы", "отчислиться из школы", "покинуть школу", "удалите меня из школы")),
    ("kb_023", ("хочу стать наставником", "стать наставником", "хочу быть наставником", "стать ментором", "хочу преподавать", "вести школу", "открыть свою школу", "стать спикером")),
    ("kb_034", ("программа наставничества", "что такое наставничество", "как работает наставничество", "расскажите про наставничество", "информация о наставничестве", "подробности о наставничестве")),
    ("kb_009", ("мой наставник", "моего наставника", "мой ментор", "у меня наставник", "наставник не отвечает", "сменить наставника", "кто мой наставник", "как найти наставника")),
    ("kb_025", ("повторно подать заявку наставнику", "заново подать заявку наставнику", "заявка к наставнику отклонена", "наставник отказал в заявке")),
    ("kb_015", ("школа уже началась", "уже идет школа", "уже идет поток", "опоздал на поток", "опоздал на школу", "уже стартовала школа", "можно ли сейчас в идущий поток")),
    ("kb_016", ("когда следующий поток", "когда новый поток", "когда следующий старт", "даты потоков", "когда следующий набор", "дата старта нового потока")),
    ("kb_002", ("не могу записаться", "не получается записаться", "не дает записаться", "не пускает в школу", "нет кнопки записи", "ошибка при записи", "не могу найти кнопку записи")),
    ("kb_033", ("как записаться", "как попасть в школу", "как вступить в школу", "как записаться на поток", "как стать учеником школы", "как начать обучение")),
    ("kb_004", ("ссылка на чат", "ссылка на телеграм", "ссылка в телеграм", "ссылка тг")),
    ("kb_021", ("добавьте в чат", "не добавили в чат", "не добавили в группу", "нет в чате школы", "нет в группе школы", "добавьте меня в телеграм", "не включили в чат")),
)


# ============================================================
# RESULT TYPES
# ============================================================


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
# TEXT HELPERS
# ============================================================


def normalize_light(text: str) -> str:
    text = (text or "").lower().strip().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    text = normalize_light(text)
    return " ".join(
        word for word in text.split()
        if word not in STOP_WORDS and len(word) > 1
    )


def contains_operator_request(text: str) -> bool:
    q = normalize_light(text)
    patterns = (
        r"\bоператор\b",
        r"\bсоедините\s+с\s+оператором\b",
        r"\bпереключите\s+на\s+оператора\b",
        r"\bхочу\s+оператора\b",
        r"\bпозовите\s+оператора\b",
        r"\bнужен\s+оператор\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _split_structural_questions(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Do not split on comma/semicolon: they often belong to one intent.
    chunks = re.split(r"(?:\?+|!+|\n+)", text)
    result: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", chunk).strip()
        if not chunk:
            continue
        # Split repeated numbered items that were in one line.
        for item in re.split(r"\s+(?=\d+[.)]\s+)", chunk):
            item = re.sub(r"^\s*\d+[.)]\s*", "", item).strip()
            if item:
                result.append(item)

    meaningful = [item for item in result if len(item.split()) >= 2]
    return meaningful[:MAX_SUBQUESTIONS] if len(meaningful) > 1 else [text]


def _split_conjunctive_question(text: str) -> list[str]:
    """Conservative split for '... и ...' only when both sides look answerable."""
    q = text.strip()
    for pattern in (r"\s+а\s+также\s+", r"\s+и еще\s+", r"\s+и\s+"):
        parts = [p.strip(" ,") for p in re.split(pattern, q, maxsplit=1, flags=re.IGNORECASE)]
        if len(parts) == 2 and all(len(p.split()) >= 3 for p in parts):
            return parts
    return [q]


def split_user_questions(text: str, max_questions: int = MAX_SUBQUESTIONS) -> list[str]:
    structural = _split_structural_questions(text)
    if len(structural) > 1:
        return structural[:max_questions]
    return _split_conjunctive_question(text)[:max_questions]


# ============================================================
# KNOWLEDGE BASE
# ============================================================


class KnowledgeBase:
    """
    Hybrid KB classifier.

    Pipeline:
        operator
          -> safe exact/fuzzy
          -> E5 phrase retrieval
          -> rule evidence (required/negative/priority)
          -> CrossEncoder rerank
          -> conflict-aware overrides
          -> TEMPLATE / CLARIFICATION / LLM / NO_MATCH

    Design principles:
      * match_phrases are positive retrieval examples.
      * description/structured guidance is context, not embedding text.
      * negative_phrases and required_phrases are deterministic evidence.
      * LLM receives only retrieved candidates.
      * weak retrieval is NOT sent blindly to the LLM.
      * multi-question input is split conservatively and each sub-question
        is classified independently.
    """

    def __init__(
        self,
        path: str = "data/knowledge_base.json",
        model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        device: str = "cpu",
        use_reranker: bool = True,
    ) -> None:
        self.entries: list[dict] = []
        self.entries_by_id: dict[str, dict] = {}
        self.phrase_records: list[dict[str, Any]] = []
        self.phrase_embeddings_matrix = np.empty((0, 0), dtype=np.float32)
        self.default_answer = "Добрый день!\nУточните, пожалуйста, подробнее ваш вопрос."

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)

        self.reranker: CrossEncoder | None = None
        if use_reranker:
            logger.info("Loading reranker model: %s", reranker_model_name)
            self.reranker = CrossEncoder(
                reranker_model_name,
                device=device,
                max_length=512,
            )

        self._load(path)
        self._build_phrase_embeddings()

        logger.info(
            "Knowledge base loaded | categories=%d | phrases=%d | reranker=%s",
            len(self.entries),
            len(self.phrase_records),
            self.reranker is not None,
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def _load(self, path: str) -> None:
        kb_path = Path(path)
        if not kb_path.exists():
            logger.error("Knowledge base not found: %s", path)
            return

        try:
            data = json.loads(kb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load knowledge base: %s", path)
            return

        if not isinstance(data, list):
            logger.error("Knowledge base root must be a JSON list")
            return

        for raw in data:
            if not isinstance(raw, dict):
                continue

            if raw.get("id") == "default":
                self.default_answer = str(raw.get("template_answer") or self.default_answer)
                continue

            entry = dict(raw)
            entry_id = str(entry.get("id") or "").strip()
            if not entry_id or entry_id in self.entries_by_id:
                continue

            for field in (
                "match_phrases",
                "negative_phrases",
                "required_phrases",
                "when_to_use",
                "when_not_to_use",
                "confused_with",
            ):
                value = entry.get(field, [])
                entry[field] = value if isinstance(value, list) else []

            clean_phrases: list[str] = []
            seen: set[str] = set()
            for phrase in entry["match_phrases"]:
                if not isinstance(phrase, str):
                    continue
                phrase = phrase.strip()
                key = normalize_light(phrase)
                if key and key not in seen:
                    seen.add(key)
                    clean_phrases.append(phrase)
            entry["match_phrases"] = clean_phrases

            # Normalize deterministic phrases. We keep original strings for LLM.
            entry["_negative_norm"] = [
                normalize_light(p) for p in entry["negative_phrases"]
                if isinstance(p, str) and normalize_light(p)
            ]
            entry["_required_norm"] = [
                normalize_light(p) for p in entry["required_phrases"]
                if isinstance(p, str) and normalize_light(p)
            ]
            entry["priority"] = int(entry.get("priority") or 0)
            entry["id"] = entry_id

            self.entries.append(entry)
            self.entries_by_id[entry_id] = entry

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    def _build_phrase_embeddings(self) -> None:
        self.phrase_records = []
        passages: list[str] = []

        for entry in self.entries:
            for index, phrase in enumerate(entry["match_phrases"]):
                self.phrase_records.append({
                    "entry_id": entry["id"],
                    "phrase_index": index,
                    "phrase": phrase,
                })
                passages.append(f"passage: {phrase}")

        if not passages:
            return

        embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.phrase_embeddings_matrix = np.asarray(embeddings, dtype=np.float32)

    # --------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------

    def get_by_id(self, kb_id: str) -> dict | None:
        return self.entries_by_id.get(str(kb_id))

    # --------------------------------------------------------
    # LEXICAL EVIDENCE
    # --------------------------------------------------------

    @staticmethod
    def _phrase_present(question_norm: str, phrase_norm: str) -> bool:
        if not phrase_norm:
            return False
        return phrase_norm in question_norm

    def _rule_evidence(self, question: str, entry: dict) -> dict[str, Any]:
        q = normalize_light(question)
        required = entry.get("_required_norm", [])
        negative = entry.get("_negative_norm", [])

        required_hits = [p for p in required if self._phrase_present(q, p)]
        required_missing = [p for p in required if p not in required_hits]
        negative_hits = [p for p in negative if self._phrase_present(q, p)]

        score_adjustment = 0.0
        if required:
            if not required_hits:
                score_adjustment -= REQUIRED_PENALTY
            elif len(required_hits) == len(required):
                score_adjustment += REQUIRED_BONUS
            else:
                missing_ratio = len(required_missing) / len(required)
                score_adjustment -= REQUIRED_PENALTY * missing_ratio

        if negative_hits:
            score_adjustment -= NEGATIVE_PENALTY * min(2, len(negative_hits))

        return {
            "required_hits": required_hits,
            "required_missing": required_missing,
            "negative_hits": negative_hits,
            "rule_adjustment": score_adjustment,
        }

    # --------------------------------------------------------
    # FUZZY
    # --------------------------------------------------------

    def _fuzzy_candidates(self, question: str) -> list[dict]:
        q_light = normalize_light(question)
        q_aggressive = normalize(question)
        results: list[dict] = []

        for entry in self.entries:
            best = None
            best_score = 0.0

            for phrase in entry.get("match_phrases", []):
                p_light = normalize_light(phrase)
                if q_light == p_light:
                    score = 100.0
                    method = "exact"
                else:
                    full_ratio = float(fuzz.ratio(q_light, p_light))
                    token_ratio = float(fuzz.token_set_ratio(q_aggressive, normalize(phrase)))
                    # Token-set may become artificially high when the question
                    # contains the phrase plus lots of extra words, so cap its
                    # influence using full-string similarity.
                    score = max(full_ratio, min(token_ratio, full_ratio + 5.0))
                    method = "fuzzy"

                if score > best_score:
                    best_score = score
                    best = (phrase, method)

            if best:
                results.append({
                    "entry": entry,
                    "score": best_score,
                    "phrase": best[0],
                    "method": best[1],
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _safe_fuzzy_match(self, fuzzy_results: list[dict], question: str) -> dict | None:
        if not fuzzy_results:
            return None
        top = fuzzy_results[0]
        evidence = self._rule_evidence(question, top["entry"])
        if evidence["negative_hits"]:
            return None

        second_score = fuzzy_results[1]["score"] if len(fuzzy_results) > 1 else 0.0
        margin = float(top["score"] - second_score)

        if top["method"] == "exact":
            return top

        if top["score"] >= FUZZY_EXACT_THRESHOLD and margin >= FUZZY_MARGIN_THRESHOLD:
            return top

        return None

    # --------------------------------------------------------
    # E5 RETRIEVAL
    # --------------------------------------------------------

    def _retrieve_candidates(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        if self.phrase_embeddings_matrix.size == 0:
            return []

        query_embedding = self.model.encode(
            f"query: {question}",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores = np.dot(self.phrase_embeddings_matrix, query_embedding)
        best_per_entry: dict[str, dict] = {}

        for index, score in enumerate(scores):
            record = self.phrase_records[index]
            entry = self.entries_by_id.get(record["entry_id"])
            if entry is None:
                continue
            item = {
                "entry": entry,
                "embedding_score": float(score),
                "best_phrase": record["phrase"],
            }
            current = best_per_entry.get(record["entry_id"])
            if current is None or item["embedding_score"] > current["embedding_score"]:
                best_per_entry[record["entry_id"]] = item

        return sorted(
            best_per_entry.values(),
            key=lambda x: x["embedding_score"],
            reverse=True,
        )[:top_k]

    # --------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _build_reranker_text(entry: dict, best_phrase: str) -> str:
        category = str(entry.get("category") or "")
        scope = str(entry.get("scope") or entry.get("description") or "")
        when_to_use = "; ".join(str(x) for x in entry.get("when_to_use", [])[:4])
        when_not = "; ".join(str(x) for x in entry.get("when_not_to_use", [])[:4])
        negatives = "; ".join(str(x) for x in entry.get("negative_phrases", [])[:6])
        return (
            f"Категория: {category}. "
            f"Область: {scope}. "
            f"Когда применять: {when_to_use}. "
            f"Когда не применять: {when_not}. "
            f"Лучший пример: {best_phrase}. "
            f"Не путать с примерами: {negatives}."
        )

    def _rerank_candidates(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        pairs = [
            (
                question,
                self._build_reranker_text(item["entry"], item["best_phrase"]),
            )
            for item in candidates
        ]

        if self.reranker is None:
            raw_scores = [item["embedding_score"] for item in candidates]
        else:
            raw_scores = self.reranker.predict(
                pairs,
                show_progress_bar=False,
            )

        ranked: list[dict] = []
        for candidate, raw in zip(candidates, raw_scores):
            raw_score = float(np.asarray(raw).reshape(-1)[0])
            rerank_score = self._sigmoid(raw_score) if self.reranker is not None else self._sigmoid(raw_score * 4.0)
            item = dict(candidate)
            item["rerank_raw"] = raw_score
            item["rerank_score"] = rerank_score

            evidence = self._rule_evidence(question, candidate["entry"])
            item.update(evidence)
            priority = int(candidate["entry"].get("priority") or 0)
            item["final_score"] = rerank_score + evidence["rule_adjustment"] + min(priority, 150) / 150.0 * PRIORITY_TIE_WEIGHT
            ranked.append(item)

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked[:RERANK_TOP_K]

    # --------------------------------------------------------
    # BUSINESS OVERRIDES
    # --------------------------------------------------------

    def _apply_entry_rules(self, question: str, ranked: list[dict]) -> list[dict]:
        q = normalize_light(question)

        def promote(entry_id: str) -> list[dict]:
            target = next((item for item in ranked if item["entry"].get("id") == entry_id), None)
            if target is None:
                return ranked
            return [target] + [item for item in ranked if item is not target]

        for entry_id, phrases in ENTRY_RULES:
            if any(normalize_light(phrase) in q for phrase in phrases):
                target = next((item for item in ranked if item["entry"].get("id") == entry_id), None)
                if target is not None and not target.get("negative_hits"):
                    return promote(entry_id)

        # Specific high-conflict contextual rules.
        if "ссылка" in q and any(x in q for x in ("чат", "телеграм", "тг")):
            return promote("kb_004")

        if "заявк" in q and "наставник" in q and any(x in q for x in ("повторно", "снова", "отклон", "отказ")):
            return promote("kb_025")

        if "наставник" in q and any(x in q for x in ("мой наставник", "моего наставника", "мой ментор", "у меня наставник", "наставник не отвечает", "сменить наставника", "кто мой наставник", "как найти наставника")):
            return promote("kb_009")

        if "наставник" in q and any(x in q for x in ("стать наставником", "стать ментором", "хочу преподавать", "вести школу")):
            return promote("kb_023")

        return ranked

    # --------------------------------------------------------
    # LLM PAYLOAD
    # --------------------------------------------------------

    @staticmethod
    def _clean_entry_for_llm(entry: dict) -> dict:
        # Deliberately pass positive/negative examples and structured guidance.
        # Internal normalized helper keys are not exposed.
        allowed = {
            "id", "category", "match_phrases", "negative_phrases",
            "required_phrases", "priority", "scope", "when_to_use",
            "when_not_to_use", "confused_with", "requires_clarification",
            "clarification_question", "template_answer", "description",
        }
        return {key: value for key, value in entry.items() if key in allowed}

    def _build_relevant_entries(self, ranked: list[dict]) -> list[dict]:
        result: list[dict] = []
        for item in ranked[:RERANK_TOP_K]:
            clean = self._clean_entry_for_llm(item["entry"])
            clean["_embedding_score"] = float(item["embedding_score"])
            clean["_rerank_score"] = float(item["rerank_score"])
            clean["_rerank_raw"] = float(item["rerank_raw"])
            clean["_final_score"] = float(item["final_score"])
            clean["_best_match_phrase"] = item["best_phrase"]
            clean["_required_hits"] = item["required_hits"]
            clean["_negative_hits"] = item["negative_hits"]
            result.append(clean)
        return result

    # --------------------------------------------------------
    # MAIN MATCH
    # --------------------------------------------------------

    def match(self, question: str) -> MatchResult:
        question = (question or "").strip()

        if not question:
            return MatchResult(
                mode=MatchMode.CLARIFICATION,
                score=0.0,
                category=None,
                template_answer=self.default_answer,
                kb_id=None,
                relevant_entries=[],
                subquestion=question,
            )

        if contains_operator_request(question):
            return MatchResult(
                mode=MatchMode.OPERATOR,
                score=1.0,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
                subquestion=question,
            )

        if not self.entries:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        # 1) Exact/very-safe fuzzy. It only fires on an extremely close FULL question.
        fuzzy = self._fuzzy_candidates(question)
        best_fuzzy = fuzzy[0] if fuzzy else None
        safe_fuzzy = self._safe_fuzzy_match(fuzzy, question)

        if safe_fuzzy:
            entry = safe_fuzzy["entry"]
            second = fuzzy[1]["score"] if len(fuzzy) > 1 else 0.0
            margin = safe_fuzzy["score"] - second
            if entry.get("requires_clarification"):
                return MatchResult(
                    mode=MatchMode.CLARIFICATION,
                    score=safe_fuzzy["score"] / 100.0,
                    category=entry.get("category"),
                    template_answer=entry.get("clarification_question"),
                    kb_id=entry.get("id"),
                    relevant_entries=[self._clean_entry_for_llm(entry)],
                    entry=entry,
                    second_score=second / 100.0,
                    margin=margin / 100.0,
                    fuzzy_score=safe_fuzzy["score"] / 100.0,
                    subquestion=question,
                )
            return MatchResult(
                mode=MatchMode.TEMPLATE,
                score=safe_fuzzy["score"] / 100.0,
                category=entry.get("category"),
                template_answer=(entry.get("template_answer") or "") + OPERATOR_HINT,
                kb_id=entry.get("id"),
                relevant_entries=[self._clean_entry_for_llm(entry)],
                entry=entry,
                second_score=second / 100.0,
                margin=margin / 100.0,
                fuzzy_score=safe_fuzzy["score"] / 100.0,
                subquestion=question,
            )

        # 2) E5 retrieval.
        retrieval = self._retrieve_candidates(question)
        if not retrieval:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        top_embedding = float(retrieval[0]["embedding_score"])
        if top_embedding < EMBEDDING_MIN_SCORE:
            return MatchResult(
                mode=MatchMode.NO_MATCH,
                score=top_embedding,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
                fuzzy_score=best_fuzzy["score"] / 100.0 if best_fuzzy else 0.0,
                embedding_score=top_embedding,
                subquestion=question,
            )

        # 3) CrossEncoder + deterministic evidence.
        ranked = self._rerank_candidates(question, retrieval)
        ranked = self._apply_entry_rules(question, ranked)
        if not ranked:
            return MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=question)

        top = ranked[0]
        top_score = float(top["final_score"])
        second_score = float(ranked[1]["final_score"]) if len(ranked) > 1 else 0.0
        margin = top_score - second_score if len(ranked) > 1 else top_score
        entry = top["entry"]
        relevant_entries = self._build_relevant_entries(ranked)

        logger.info(
            "KB decision | kb_id=%s | rerank=%.4f | final=%.4f | margin=%.4f | embedding=%.4f | required=%s | negative=%s",
            entry.get("id"),
            top["rerank_score"],
            top_score,
            margin,
            top_embedding,
            top["required_hits"],
            top["negative_hits"],
        )

        if entry.get("requires_clarification"):
            return MatchResult(
                mode=MatchMode.CLARIFICATION,
                score=top_score,
                category=entry.get("category"),
                template_answer=entry.get("clarification_question"),
                kb_id=entry.get("id"),
                relevant_entries=relevant_entries,
                entry=entry,
                second_score=second_score,
                margin=margin,
                fuzzy_score=best_fuzzy["score"] / 100.0 if best_fuzzy else 0.0,
                embedding_score=top_embedding,
                subquestion=question,
            )

        # 4) High-confidence deterministic template.
        if (
            top["rerank_score"] >= RERANK_AUTO_THRESHOLD
            and margin >= RERANK_MARGIN_THRESHOLD
            and not top["negative_hits"]
            and top_embedding >= EMBEDDING_MIN_SCORE
        ):
            return MatchResult(
                mode=MatchMode.TEMPLATE,
                score=top_score,
                category=entry.get("category"),
                template_answer=(entry.get("template_answer") or "") + OPERATOR_HINT,
                kb_id=entry.get("id"),
                relevant_entries=relevant_entries,
                entry=entry,
                second_score=second_score,
                margin=margin,
                fuzzy_score=best_fuzzy["score"] / 100.0 if best_fuzzy else 0.0,
                embedding_score=top_embedding,
                subquestion=question,
            )

        # 5) Plausible but ambiguous -> constrained LLM.
        if (
            top["rerank_score"] >= RERANK_LLM_THRESHOLD
            and top_embedding >= LLM_MIN_EMBEDDING_SCORE
            and margin >= 0.01
        ):
            return MatchResult(
                mode=MatchMode.LLM,
                score=top_score,
                category=entry.get("category"),
                template_answer=None,
                kb_id=entry.get("id"),
                relevant_entries=relevant_entries,
                entry=entry,
                second_score=second_score,
                margin=margin,
                fuzzy_score=best_fuzzy["score"] / 100.0 if best_fuzzy else 0.0,
                embedding_score=top_embedding,
                subquestion=question,
            )

        # 6) Weak/noisy retrieval -> operator instead of hallucinating a route.
        return MatchResult(
            mode=MatchMode.NO_MATCH,
            score=top_score,
            category=None,
            template_answer=None,
            kb_id=None,
            relevant_entries=relevant_entries,
            entry=None,
            second_score=second_score,
            margin=margin,
            fuzzy_score=best_fuzzy["score"] / 100.0 if best_fuzzy else 0.0,
            embedding_score=top_embedding,
            subquestion=question,
        )

    # --------------------------------------------------------
    # MULTI-QUESTION
    # --------------------------------------------------------

    def match_many(self, question: str) -> list[MatchResult]:
        question = (question or "").strip()
        parts = split_user_questions(question)
        if len(parts) <= 1:
            return [self.match(question)]

        results = [self.match(part) for part in parts]

        # If a conjunction split created a weak/no-match fragment, prefer the
        # original message as one classification rather than inventing two intents.
        if any(r.mode == MatchMode.NO_MATCH for r in results):
            whole = self.match(question)
            if whole.mode != MatchMode.NO_MATCH:
                return [whole]

        return results[:MAX_SUBQUESTIONS]

    # --------------------------------------------------------
    # DEBUG / PUBLIC RETRIEVAL
    # --------------------------------------------------------

    def get_relevant_entries(self, question: str, top_n: int = RERANK_TOP_K) -> list[dict]:
        candidates = self._retrieve_candidates(question, max(RETRIEVAL_TOP_K, top_n))
        ranked = self._apply_entry_rules(question, self._rerank_candidates(question, candidates))
        return self._build_relevant_entries(ranked[:top_n])

    def debug_match(self, question: str) -> dict:
        fuzzy = self._fuzzy_candidates(question)
        retrieval = self._retrieve_candidates(question)
        ranked = self._apply_entry_rules(question, self._rerank_candidates(question, retrieval))
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
                "fuzzy_score": decision.fuzzy_score,
                "embedding_score": decision.embedding_score,
            },
            "fuzzy": [
                {
                    "kb_id": x["entry"]["id"],
                    "score": x["score"],
                    "method": x["method"],
                    "phrase": x["phrase"],
                }
                for x in fuzzy[:5]
            ],
            "retrieval": [
                {
                    "kb_id": x["entry"]["id"],
                    "embedding_score": x["embedding_score"],
                    "phrase": x["best_phrase"],
                }
                for x in retrieval[:RETRIEVAL_TOP_K]
            ],
            "reranked": [
                {
                    "kb_id": x["entry"]["id"],
                    "embedding_score": x["embedding_score"],
                    "rerank_score": x["rerank_score"],
                    "final_score": x["final_score"],
                    "required_hits": x["required_hits"],
                    "negative_hits": x["negative_hits"],
                    "phrase": x["best_phrase"],
                }
                for x in ranked
            ],
        }