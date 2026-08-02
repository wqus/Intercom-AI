import json
import logging
import re

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger("knowledge_base")

NO_MATCH_THRESHOLD = 20
DIRECT_THRESHOLD = 90
GAP_THRESHOLD = 25
TOP_N = 5

STOP_WORDS = {
    "не",
    "могу",
    "как",
    "хочу",
    "что",
    "это",
    "мне",
    "у",
    "меня",
    "мой",
    "моя",
    "моё",
    "можно",
    "нужно",
    "надо",
    "есть",
    "был",
    "была",
    "буду",
    "вы",
    "я",
    "он",
    "она",
    "пожалуйста",
    "помогите",
    "скажите",
    "подскажите",
}

OPERATOR_HINT = (
    "\n\n---\n"
    "Если этот ответ не помог решить ваш вопрос, просто напишите «оператор». "
    "Мы повторно рассмотрим ваше обращение и ответим вам в этом чате."
)


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

def normalize(text: str) -> str:
    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    words = [
        word
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 1
    ]

    return " ".join(words).strip()


def calculate_phrase_score(
        question_norm: str,
        phrase_norm: str,
) -> float:
    return max(
        fuzz.token_set_ratio(question_norm, phrase_norm),
        fuzz.partial_ratio(question_norm, phrase_norm),
    )


def best_score(
        question_norm: str,
        phrases: list[str],
) -> float:
    if not phrases:
        return 0.0

    best = 0.0

    for phrase in phrases:

        phrase_norm = normalize(phrase)

        if not phrase_norm:
            continue

        score = calculate_phrase_score(
            question_norm,
            phrase_norm,
        )

        if score > best:
            best = score

    return best


class KnowledgeBase:

    def __init__(
            self,
            path: str = "data/knowledge_base.json",
    ) -> None:

        self.entries: list[dict] = []

        self.default_answer = (
            "Добрый день!\n"
            "Уточните, пожалуйста, подробнее ваш вопрос."
        )

        self._load(path)

    def _load(
            self,
            path: str,
    ) -> None:

        kb_path = Path(path)

        if not kb_path.exists():
            logger.error(
                "Knowledge base not found: %s",
                path,
            )
            return

        with open(
                kb_path,
                encoding="utf-8",
        ) as f:
            data = json.load(f)

        for entry in data:

            if entry["id"] == "default":
                self.default_answer = entry["template_answer"]
                continue

            self.entries.append(entry)

        logger.info(
            "Knowledge base loaded: %s entries",
            len(self.entries),
        )

    def get_by_id(self, kb_id: str) -> dict | None:
        return next((e for e in self.entries if e["id"] == kb_id), None)

    def get_scored_entries(
            self,
            question: str,
    ) -> list[tuple[float, dict]]:

        question_norm = normalize(question)

        scored = []

        for entry in self.entries:
            score = best_score(
                question_norm,
                entry.get("match_phrases", []),
            )

            scored.append((score, entry))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return scored

    def match(
            self,
            question: str,
    ) -> MatchResult:

        if not self.entries:
            return MatchResult(
                mode=MatchMode.NO_MATCH,
                score=0,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )
        question_norm = normalize(question)

        if "оператор" in question_norm:
            return MatchResult(
                mode=MatchMode.OPERATOR,
                score=100,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )

        scored = self.get_scored_entries(question)

        top_score, top_entry = scored[0]

        top2_score = (
            scored[1][0]
            if len(scored) > 1
            else 0.0
        )

        gap = top_score - top2_score

        logger.info(
            "KB match | top1=%.1f | top2=%.1f | gap=%.1f | category=%s",
            top_score,
            top2_score,
            gap,
            top_entry.get("category"),
        )

        if top_score < NO_MATCH_THRESHOLD:
            return MatchResult(
                mode=MatchMode.NO_MATCH,
                score=top_score,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )

        if (
                top_score >= DIRECT_THRESHOLD
        ):
            if top_entry.get("requires_clarification"):
                return MatchResult(
                    mode=MatchMode.CLARIFICATION,
                    score=top_score,
                    category=top_entry["category"],
                    template_answer=top_entry.get("clarification_question"),
                    kb_id=top_entry["id"],
                    relevant_entries=[top_entry],
                    entry=top_entry,
                )

            return MatchResult(
                mode=MatchMode.TEMPLATE,
                score=top_score,
                category=top_entry["category"],
                template_answer=top_entry["template_answer"] + OPERATOR_HINT,
                kb_id=top_entry["id"],
                relevant_entries=[top_entry],
                entry=top_entry,  # NEW
            )

        relevant_entries = [
            entry
            for score, entry in scored[:TOP_N]
            if score >= NO_MATCH_THRESHOLD
        ]

        return MatchResult(
            mode=MatchMode.LLM,
            score=top_score,
            category=top_entry["category"],
            template_answer=None,
            kb_id=top_entry["id"],
            relevant_entries=relevant_entries,
        )

    def get_relevant_entries(
            self,
            question: str,
            top_n: int = TOP_N,
    ) -> list[dict]:

        scored = self.get_scored_entries(question)

        return [
            entry
            for score, entry in scored[:top_n]
            if score >= NO_MATCH_THRESHOLD
        ]
