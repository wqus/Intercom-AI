from __future__ import annotations

from app.services.knowledge_base import KnowledgeBase, _query_quality


CASES = [
    ("Хотел бы стать наставников", "kb_023"),
    ("Хотел бы стать наставников в будущем, это возможно", "kb_023"),
    ("Не получается записаться", "kb_002"),
    ("Как записаться в школу?", "kb_033"),
    ("Как записаться на обучение?", "kb_035"),
    ("Как попасть на обучение?", "kb_035"),
    ("Когда начало следующего потока?", "kb_016"),
    ("Не принимает мои данные", "kb_001"),
    ("Я новичок, установила приложение, зайти не получается, не принимает мои данные", "kb_001"),
    ("Хочу перейти в другую школу", "kb_006"),
    ("Меня исключили из школы", "kb_007"),
    ("Хочу восстановиться в школе", "kb_008"),
    ("Не работает инвайт-код", "kb_013"),
    ("Не работает QR-код", "kb_014"),
    ("Меня не добавили в чат", "kb_021"),
    ("Где ссылка на чат?", "kb_004"),
]


def _bare_kb() -> KnowledgeBase:
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {kb_id: {} for _, kb_id in CASES}
    return kb


def test_high_precision_rules_handle_real_production_failures() -> None:
    kb = _bare_kb()
    for question, expected in CASES:
        assert kb._hard_rule_match(question) == expected, question


def test_query_quality_cannot_block_short_but_valid_deterministic_intent() -> None:
    question = "Хотел бы стать наставников"
    quality = _query_quality(question)
    assert not quality.valid
    assert _bare_kb()._hard_rule_match(question) == "kb_023"
