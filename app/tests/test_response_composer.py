from app.domain.response_composer import ResponseComposer, clean_business_answer
from app.infrastructure.ai.knowledge_base import KnowledgeBase


def test_clean_business_answer_removes_legacy_wrappers():
    text = (
        "Добрый день!\n\nОтвет на вопрос.\n\n---\n"
        "Если этот ответ не помог решить ваш вопрос, просто напишите «оператор». "
        "Мы повторно рассмотрим ваше обращение и ответим вам в этом чате."
    )
    assert clean_business_answer(text) == "Ответ на вопрос."
    new_footer = ResponseComposer.final_answer(["Ответ на вопрос."])
    assert new_footer.endswith("ваш вопрос будет передан оператору.")


def test_final_answer_has_one_greeting_and_one_footer():
    text = ResponseComposer.final_answer(["Добрый день!\nОтвет 1", "Добрый день!\nОтвет 2"])
    assert text.count("Добрый день!") == 1
    assert text.count("Если этот ответ не помог") == 1
    assert "Ответ 1" in text and "Ответ 2" in text


def test_choice_has_no_final_operator_footer():
    text = ResponseComposer.choice(["Первый", "Второй", "Третий"])
    assert text.startswith("Добрый день!")
    assert "Если этот ответ не помог" not in text
    assert "1. Первый" in text


def test_current_kb_answers_are_business_content_only():
    # This is intentionally a lightweight data contract check; model loading is
    # not needed to verify the serialized KB presentation contract.
    import json
    from pathlib import Path
    data = json.loads(Path("data/knowledge_base.json").read_text(encoding="utf-8"))
    for entry in data:
        answer = str(entry.get("template_answer") or "")
        assert not answer.lower().startswith("добрый день")
        assert "Если этот ответ не помог" not in answer


def test_choice_offers_operator_fallback():
    text = ResponseComposer.choice(["Первый", "Второй", "Третий"])
    assert "Если ни один вариант не подходит" in text
    assert "«оператор»" in text
    assert "передан оператору" in text
