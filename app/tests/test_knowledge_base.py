
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.services.knowledge_base import (
    KnowledgeBase,
    MatchMode,
    normalize,
)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]   # корень проекта
KB_PATH = ROOT_DIR / "data" / "knowledge_base.json"


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    assert KB_PATH.exists(), f"Knowledge base not found: {KB_PATH}"

    return KnowledgeBase(
        path=str(KB_PATH),
        device="cpu",
        use_reranker=False,
    )


# ============================================================
# BASIC KNOWN INTENTS
# ============================================================

KNOWN_INTENTS = [
    # Общий вход в обучение
    ("Как попасть на обучение?", "kb_035"),
    ("Хочу начать учиться у вас", "kb_035"),
    ("Что нужно чтобы попасть на обучение?", "kb_035"),
    ("Я хочу учиться в Терре, с чего начать?", "kb_035"),
    ("Хочу попасть к вам на обучение, как это сделать?", "kb_035"),
    ("Как стать участником обучения?", "kb_035"),

    # Запись именно в школу
    ("Как записаться в школу?", "kb_033"),
    ("Где в приложении выбрать школу?", "kb_033"),
    ("Как попасть именно в школу?", "kb_033"),
    ("Где выбрать школу в приложении?", "kb_033"),
    ("А где тут школу выбрать?", "kb_033"),

    # Ошибка записи
    ("Не могу записаться в школу", "kb_002"),
    ("Не получается записаться в школу", "kb_002"),
    ("Кнопки записи нет", "kb_002"),
    ("Не дает записаться", "kb_002"),

    # Наставник
    ("Где мой наставник?", "kb_009"),
    ("Наставник не отвечает", "kb_009"),
    ("Кто мой ментор?", "kb_009"),
    ("Мне наставника не дали", "kb_009"),

    # Инвайт
    ("Не работает инвайт-код", "kb_013"),
    ("Код приглашения не подходит", "kb_013"),
    ("Инвайт не работает", "kb_013"),

    # QR
    ("Не работает QR-код", "kb_014"),
    ("QR-код не сканируется", "kb_014"),
    ("QR не читается", "kb_014"),

    # Telegram / ссылка
    ("Где ссылка на чат?", "kb_004"),
    ("Не открывается ссылка на Telegram", "kb_004"),
    ("Ссылка на чат не работает", "kb_004"),

    # Не добавили в чат
    ("Меня не добавили в чат", "kb_021"),
    ("Нет меня в группе школы", "kb_021"),
    ("Добавьте меня в Telegram", "kb_021"),

    # Выход / переход / исключение / восстановление
    ("Хочу выйти из школы", "kb_005"),
    ("Хочу сменить школу", "kb_006"),
    ("Меня исключили из школы", "kb_007"),
    ("Хочу восстановиться в школе", "kb_008"),
]


@pytest.mark.parametrize(
    ("question", "expected_kb"),
    KNOWN_INTENTS,
    ids=[f"{kb_id}:{question}" for question, kb_id in KNOWN_INTENTS],
)
def test_known_intents(
    kb: KnowledgeBase,
    question: str,
    expected_kb: str,
) -> None:
    result = kb.match(question)

    assert result.kb_id == expected_kb, (
        f"\nQuestion: {question}"
        f"\nExpected: {expected_kb}"
        f"\nGot: {result.kb_id}"
        f"\nMode: {result.mode}"
        f"\nScore: {result.score}"
        f"\nSecond score: {result.second_score}"
        f"\nMargin: {result.margin}"
    )


# ============================================================
# CONFLICT / HARD-NEGATIVE CASES
# These are the most important tests for your classifier.
# ============================================================

CONFLICT_CASES = [
    # kb_033 vs kb_035
    ("Как записаться в школу?", "kb_033"),
    ("Как попасть на обучение?", "kb_035"),
    ("Как начать обучение?", "kb_035"),
    ("Где выбрать школу в приложении?", "kb_033"),
    ("Хочу попасть в школу", "kb_033"),
    ("Хочу попасть к вам на обучение", "kb_035"),
    ("Как записаться на обучение?", "kb_035"),

    # kb_002 vs kb_033
    ("Как записаться в школу?", "kb_033"),
    ("Не могу записаться в школу", "kb_002"),
    ("Не получается записаться", "kb_002"),
    ("Кнопки записи нет", "kb_002"),
    ("Как оформить запись в школу?", "kb_033"),

    # kb_004 vs kb_021
    ("Где ссылка на чат?", "kb_004"),
    ("Ссылка на чат не работает", "kb_004"),
    ("Не открывается Telegram-чат по ссылке", "kb_004"),
    ("Меня не добавили в чат", "kb_021"),
    ("Нет меня в группе школы", "kb_021"),
    ("Добавьте меня в чат", "kb_021"),

    # kb_013 vs kb_014
    ("Не работает инвайт-код", "kb_013"),
    ("Инвайт не работает", "kb_013"),
    ("Код приглашения не подходит", "kb_013"),
    ("Не работает QR-код", "kb_014"),
    ("QR-код не сканируется", "kb_014"),
    ("QR не читается", "kb_014"),

    # kb_005 / kb_006 / kb_007 / kb_008
    ("Хочу выйти из школы", "kb_005"),
    ("Хочу сменить школу", "kb_006"),
    ("Хочу перейти в другую школу", "kb_006"),
    ("Меня исключили из школы", "kb_007"),
    ("Меня отчислили", "kb_007"),
    ("Хочу восстановиться в школе", "kb_008"),
    ("Хочу вернуться в школу", "kb_008"),
]


@pytest.mark.parametrize(
    ("question", "expected_kb"),
    CONFLICT_CASES,
    ids=[f"{kb_id}:{question}" for question, kb_id in CONFLICT_CASES],
)
def test_conflicting_intents(
    kb: KnowledgeBase,
    question: str,
    expected_kb: str,
) -> None:
    result = kb.match(question)

    assert result.kb_id == expected_kb, (
        f"\nCONFLICT FAILURE"
        f"\nQuestion: {question}"
        f"\nExpected: {expected_kb}"
        f"\nGot: {result.kb_id}"
        f"\nMode: {result.mode}"
        f"\nScore: {result.score}"
        f"\nSecond score: {result.second_score}"
        f"\nMargin: {result.margin}"
    )


# ============================================================
# GARBAGE / UNKNOWN / OUT-OF-DOMAIN
# ============================================================

GARBAGE_CASES = [
    "Очшыоалыдажыжаэв к",
    "asdfghjkl",
    "qwertyuiop",
    "123123123",
    "фывапролдж",
    "!!! ???",
    "абвгд",
]


@pytest.mark.parametrize("question", GARBAGE_CASES)
def test_garbage_is_not_classified(
    kb: KnowledgeBase,
    question: str,
) -> None:
    result = kb.match(question)

    assert result.kb_id is None, (
        f"\nGarbage was classified!"
        f"\nQuestion: {question}"
        f"\nGot: {result.kb_id}"
        f"\nMode: {result.mode}"
        f"\nScore: {result.score}"
    )


# ============================================================
# NORMALIZATION SAFETY
# ============================================================

def test_negation_is_preserved() -> None:
    normalized = normalize("не могу записаться")

    assert "не" in normalized
    assert "могу" in normalized
    assert "записаться" in normalized


@pytest.mark.parametrize(
    "text",
    [
        "не могу",
        "нет кнопки",
        "без доступа",
        "уже началась школа",
    ],
)
def test_semantically_important_words_are_not_deleted(text: str) -> None:
    normalized = normalize(text)

    important_words = {
        "не",
        "нет",
        "без",
        "уже",
    }

    # At least one important token from the original text must survive.
    assert any(word in normalized.split() for word in important_words if word in text.lower())


# ============================================================
# KB STRUCTURE
# ============================================================

def test_kb_file_exists() -> None:
    assert KB_PATH.exists(), f"Missing KB: {KB_PATH}"


def test_kb_loaded(kb: KnowledgeBase) -> None:
    assert kb.entries, "Knowledge base is empty"
    assert kb.entries_by_id, "entries_by_id is empty"


def test_kb_has_unique_ids(kb: KnowledgeBase) -> None:
    ids = [entry["id"] for entry in kb.entries]

    assert len(ids) == len(set(ids)), "Duplicate KB IDs detected"


def test_kb_entries_have_required_fields(kb: KnowledgeBase) -> None:
    required_fields = {
        "id",
        "category",
        "match_phrases",
        "template_answer",
        "description",
    }

    for entry in kb.entries:
        missing = required_fields - set(entry.keys())
        assert not missing, (
            f"KB entry {entry.get('id')} is missing fields: {sorted(missing)}"
        )


def test_kb_conflicts_reference_existing_ids(kb: KnowledgeBase) -> None:
    known_ids = set(kb.entries_by_id)

    for entry in kb.entries:
        for conflict_id in entry.get("conflict_ids", []):
            assert conflict_id in known_ids, (
                f"{entry['id']} references unknown conflict ID {conflict_id}"
            )


# ============================================================
# MULTI-QUESTION
# ============================================================

MULTI_QUESTION_CASES = [
    (
        "Как попасть на обучение? Где записаться в школу?",
        {"kb_035", "kb_033"},
    ),
]


@pytest.mark.parametrize(
    ("question", "expected_ids"),
    MULTI_QUESTION_CASES,
)
def test_multi_question(
    kb: KnowledgeBase,
    question: str,
    expected_ids: set[str],
) -> None:
    results = kb.match_many(question)

    actual_ids = {
        result.kb_id
        for result in results
        if result.kb_id is not None
    }

    assert expected_ids.issubset(actual_ids), (
        f"\nQuestion: {question}"
        f"\nExpected IDs: {expected_ids}"
        f"\nActual IDs: {actual_ids}"
    )


# ============================================================
# BENCHMARK
# This is intentionally inside the same file.
# Add real user questions here as they appear in Intercom.
# ============================================================

BENCHMARK = [
    # kb_035
    ("Как попасть на обучение?", "kb_035"),
    ("Хочу начать учиться у вас", "kb_035"),
    ("Что нужно чтобы попасть на обучение?", "kb_035"),
    ("Я хочу учиться в Терре, с чего начать?", "kb_035"),
    ("Хочу попасть к вам на обучение, как это сделать?", "kb_035"),
    ("Как стать участником обучения?", "kb_035"),

    # kb_033
    ("Как записаться в школу?", "kb_033"),
    ("Где в приложении выбрать школу?", "kb_033"),
    ("Где выбрать школу?", "kb_033"),
    ("Как попасть именно в школу?", "kb_033"),

    # kb_002
    ("Не могу записаться в школу", "kb_002"),
    ("Не получается записаться", "kb_002"),
    ("Кнопки записи нет", "kb_002"),
    ("Не дает записаться", "kb_002"),

    # kb_009
    ("Где мой наставник?", "kb_009"),
    ("Наставник не отвечает", "kb_009"),
    ("Кто мой ментор?", "kb_009"),

    # kb_013
    ("Не работает инвайт-код", "kb_013"),
    ("Инвайт не работает", "kb_013"),

    # kb_014
    ("Не работает QR-код", "kb_014"),
    ("QR не сканируется", "kb_014"),

    # kb_004
    ("Где ссылка на чат?", "kb_004"),
    ("Ссылка на чат не работает", "kb_004"),

    # kb_021
    ("Меня не добавили в чат", "kb_021"),
    ("Нет меня в группе школы", "kb_021"),

    # unknown
    ("Очшыоалыдажыжаэв к", "unknown"),
    ("asdfghjkl", "unknown"),
]


def run_benchmark(kb: KnowledgeBase) -> dict:
    total = len(BENCHMARK)
    correct = 0
    failures: list[dict] = []
    confusion = Counter()

    per_intent_total = Counter()
    per_intent_correct = Counter()

    for question, expected in BENCHMARK:
        per_intent_total[expected] += 1

        result = kb.match(question)
        predicted = result.kb_id if result.kb_id is not None else "unknown"

        if predicted == expected:
            correct += 1
            per_intent_correct[expected] += 1
        else:
            failures.append(
                {
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "mode": result.mode.value,
                    "score": result.score,
                    "second_score": result.second_score,
                    "margin": result.margin,
                }
            )

            if expected != predicted:
                confusion[(expected, predicted)] += 1

    accuracy = correct / total if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "failures": failures,
        "confusion": confusion,
        "per_intent_total": per_intent_total,
        "per_intent_correct": per_intent_correct,
    }


def print_benchmark_report(report: dict) -> None:
    print("\n")
    print("=" * 72)
    print("KNOWLEDGE BASE BENCHMARK")
    print("=" * 72)

    print(f"Total:    {report['total']}")
    print(f"Correct:  {report['correct']}")
    print(f"Wrong:    {report['wrong']}")
    print(f"Accuracy: {report['accuracy']:.2%}")

    print("\nPer intent:")
    print("-" * 72)

    totals: Counter = report["per_intent_total"]
    corrects: Counter = report["per_intent_correct"]

    for intent in sorted(totals):
        total = totals[intent]
        correct = corrects[intent]
        score = correct / total if total else 0.0

        print(
            f"{intent:12} "
            f"{correct:3}/{total:<3} "
            f"{score:7.2%}"
        )

    if report["confusion"]:
        print("\nConfusion:")
        print("-" * 72)

        for (expected, predicted), count in report["confusion"].most_common():
            print(
                f"{expected:12} -> {predicted:12} "
                f"{count} time(s)"
            )

    if report["failures"]:
        print("\nFailures:")
        print("-" * 72)

        for failure in report["failures"]:
            print(
                f"\nQuestion:    {failure['question']}"
                f"\nExpected:    {failure['expected']}"
                f"\nPredicted:   {failure['predicted']}"
                f"\nMode:        {failure['mode']}"
                f"\nScore:       {failure['score']:.4f}"
                f"\nSecond:      {failure['second_score']:.4f}"
                f"\nMargin:      {failure['margin']:.4f}"
            )

    print("=" * 72)


def test_benchmark(
    kb: KnowledgeBase,
    request: pytest.FixtureRequest,
) -> None:
    report = run_benchmark(kb)
    print_benchmark_report(report)

    # Do NOT demand 100% here.
    #
    # This benchmark is a monitoring tool, not a hard quality gate.
    # The deterministic tests above are the hard gate.
    #
    # We only prevent a catastrophic regression.
    min_accuracy = 0.75

    assert report["accuracy"] >= min_accuracy, (
        f"\nBenchmark accuracy dropped below {min_accuracy:.0%}: "
        f"{report['accuracy']:.2%}"
    )


# ============================================================
# OPTIONAL STANDALONE BENCHMARK MODE
#
# Run:
#   python app/tests/test_knowledge_base.py
#
# or simply use:
#   python -m pytest app/tests/test_knowledge_base.py -v -s
# ============================================================

if __name__ == "__main__":
    kb_instance = KnowledgeBase(
        path=str(KB_PATH),
        device="cpu",
        use_reranker=False,
    )

    benchmark_report = run_benchmark(kb_instance)
    print_benchmark_report(benchmark_report)
