from app.infrastructure.ai.knowledge_base import KnowledgeBase


def test_parse_numeric_choice():
    assert KnowledgeBase.parse_category_choice("2", max_index=3) == [2]
    assert KnowledgeBase.parse_category_choice("1 и 3", max_index=3) == [1, 3]
    assert KnowledgeBase.parse_category_choice("вариант 2", max_index=3) == [2]


def test_parse_russian_ordinal_choice():
    assert KnowledgeBase.parse_category_choice("мне подходит первый", max_index=3) == [1]
    assert KnowledgeBase.parse_category_choice("беру первый и третий", max_index=3) == [1, 3]
    assert KnowledgeBase.parse_category_choice("второй вариант", max_index=3) == [2]


def test_parse_rejects_out_of_range_numbers():
    assert KnowledgeBase.parse_category_choice("2026", max_index=3) == []
    assert KnowledgeBase.parse_category_choice("вариант 4", max_index=3) == []
