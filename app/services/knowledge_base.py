import json
import logging
import re
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("knowledge_base")

# Параметры для эмбеддингов
SIMILARITY_THRESHOLD = 0.4  # Минимальное сходство для "совпадения"
TOP_N = 3  # Сколько категорий отправлять в LLM

# Стоп-слова для шума
STOP_WORDS = {
    "не", "могу", "как", "хочу", "что", "это", "мне", "у", "меня",
    "мой", "моя", "моё", "можно", "нужно", "надо", "есть", "был",
    "была", "буду", "вы", "я", "он", "она", "пожалуйста", "помогите",
    "скажите", "подскажите",
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
    score: float  # Для эмбеддингов это similarity (0-1)
    category: str | None
    template_answer: str | None
    kb_id: str | None
    relevant_entries: list[dict]
    entry: dict | None = None


def normalize(text: str) -> str:
    """Нормализация текста для поиска оператора и шума"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [word for word in text.split() if word not in STOP_WORDS and len(word) > 1]
    return " ".join(words).strip()


class KnowledgeBase:
    def __init__(
            self,
            path: str = "data/knowledge_base.json",
            model_name: str = "intfloat/multilingual-e5-small",
            # ИЛИ для русского: "alphaedge-ai/Qwen3-Embedding-rus-16384"
            device: str = "cpu",  # или "cuda" если есть GPU
    ) -> None:
        self.entries: list[dict] = []
        self.embeddings: list[np.ndarray] = []  # Векторы для каждой категории
        self.default_answer = (
            "Добрый день!\nУточните, пожалуйста, подробнее ваш вопрос."
        )

        # Загружаем модель эмбеддингов
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        # Загружаем базу знаний и создаем эмбеддинги
        self._load(path)
        self._build_embeddings()

        logger.info(f"Knowledge base loaded: {len(self.entries)} entries with embeddings")

    def _load(self, path: str) -> None:
        """Загружает JSON с базой знаний"""
        kb_path = Path(path)

        if not kb_path.exists():
            logger.error(f"Knowledge base not found: {path}")
            return

        with open(kb_path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            if entry["id"] == "default":
                self.default_answer = entry["template_answer"]
                continue

            if "description" not in entry:
                logger.warning(
                    f"Entry {entry.get('id')} ({entry.get('category')}) has no description field"
                )

            self.entries.append(entry)

    def _build_embeddings(self) -> None:
        """Создает эмбеддинги для всех категорий на основе description"""
        self.embeddings = []

        for entry in self.entries:
            # Берем category + description как текст для эмбеддинга
            text = f"{entry['category']} {entry['description']}"

            # Генерируем вектор
            embedding = self.model.encode(text, normalize_embeddings=True)
            self.embeddings.append(embedding)

            # Сохраняем эмбеддинг в entry для кэширования
            entry['_embedding'] = embedding

        # Преобразуем список в матрицу для быстрых вычислений
        if self.embeddings:
            self.embeddings_matrix = np.vstack(self.embeddings)
        else:
            self.embeddings_matrix = np.array([])

    def get_by_id(self, kb_id: str) -> dict | None:
        """Возвращает категорию по ID"""
        return next((e for e in self.entries if e["id"] == kb_id), None)

    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Вычисляет косинусное сходство между запросом и всеми категориями"""
        if self.embeddings_matrix.size == 0:
            return np.array([])

        # query_embedding уже нормализован (normalize_embeddings=True)
        # embeddings_matrix тоже нормализован
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        return similarities

    def match(self, question: str) -> MatchResult:
        """Главный метод для поиска категории по вопросу"""

        # 1. Проверка на оператора
        question_norm = normalize(question)
        if "оператор" in question_norm:
            return MatchResult(
                mode=MatchMode.OPERATOR,
                score=1.0,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )

        # 2. Если нет категорий — NO_MATCH
        if not self.entries:
            return MatchResult(
                mode=MatchMode.NO_MATCH,
                score=0,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )

        # 3. Генерируем эмбеддинг для вопроса
        query_embedding = self.model.encode(question, normalize_embeddings=True)

        # 4. Вычисляем сходство со всеми категориями
        similarities = self.cosine_similarity(query_embedding)

        # 5. Сортируем по убыванию сходства
        scored = sorted(
            [
                (similarities[i], self.entries[i])
                for i in range(len(self.entries))
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        top_score, top_entry = scored[0]

        # 6. Логируем результат
        top2_score = scored[1][0] if len(scored) > 1 else 0.0
        logger.info(
            f"KB match (embeddings) | top1={top_score:.3f} | top2={top2_score:.3f} | "
            f"category={top_entry.get('category')}"
        )

        # 7. Если сходство ниже порога — NO_MATCH
        if top_score < SIMILARITY_THRESHOLD:
            return MatchResult(
                mode=MatchMode.NO_MATCH,
                score=top_score,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=[],
            )

        # 8. Проверка на уточнение (clarification)
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

        # 9. Если сходство очень высокое и отрыв большой — отдаем шаблон
        # (только для очевидных случаев)
        if top_score > 0.85 and (top_score - top2_score) > 0.3:
            return MatchResult(
                mode=MatchMode.TEMPLATE,
                score=top_score,
                category=top_entry["category"],
                template_answer=top_entry["template_answer"] + OPERATOR_HINT,
                kb_id=top_entry["id"],
                relevant_entries=[top_entry],
                entry=top_entry,
            )

        # 10. Иначе — отправляем в LLM с ТОП-3 категориями
        # ❗ ВАЖНО: УДАЛЯЕМ match_phrases перед отправкой!
        relevant_entries = []
        for score, entry in scored[:TOP_N]:
            if score >= SIMILARITY_THRESHOLD:
                # Создаем копию и удаляем match_phrases
                clean_entry = entry.copy()
                clean_entry.pop("match_phrases", None)
                clean_entry.pop("_embedding", None)  # Удаляем служебное поле
                relevant_entries.append(clean_entry)

        return MatchResult(
            mode=MatchMode.LLM,
            score=top_score,
            category=top_entry["category"],
            template_answer=None,
            kb_id=top_entry["id"],
            relevant_entries=relevant_entries,  # БЕЗ match_phrases!
        )

    def get_relevant_entries(self, question: str, top_n: int = TOP_N) -> list[dict]:
        """Возвращает ТОП-N категорий по смыслу (без match_phrases)"""
        query_embedding = self.model.encode(question, normalize_embeddings=True)
        similarities = self.cosine_similarity(query_embedding)

        scored = sorted(
            [
                (similarities[i], self.entries[i])
                for i in range(len(self.entries))
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
        for score, entry in scored[:top_n]:
            if score >= SIMILARITY_THRESHOLD:
                clean_entry = entry.copy()
                clean_entry.pop("match_phrases", None)
                clean_entry.pop("_embedding", None)
                result.append(clean_entry)

        return result