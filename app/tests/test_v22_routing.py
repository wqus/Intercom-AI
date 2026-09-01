"""v22 routing regressions: high-risk, Top-3 visibility, inbound logging path."""
from __future__ import annotations

import asyncio
import logging
import json
from pathlib import Path

import pytest

from app.application.decision_pipeline import DecisionPipeline
from app.domain.enums import Channel, Decision, EscalationReason
from app.domain.models import InboundMessage
from app.domain.policy import high_risk_reply
from app.infrastructure.ai.knowledge_base import (
    MatchMode,
    MatchResult,
    KnowledgeBase,
    detect_high_risk,
)


class _FakeLLM:
    async def classify(self, *_args, **_kwargs):
        raise AssertionError("LLM must not be called")


def test_detect_high_risk_refund():
    assert detect_high_risk("Прошу вернуть денежные средства, оплатила и не смогла войти") == "refund"
    assert detect_high_risk("Хочу возврат денег после оплаты") == "refund"


def test_detect_high_risk_personal_data():
    assert detect_high_risk("Прошу удалить мои персональные данные из базы") == "personal_data"
    assert detect_high_risk("Отзываю согласие, удалите мои данные") == "personal_data"


def test_detect_high_risk_complaint():
    assert detect_high_risk("Одна хамоватая особа, прошу принять меры") == "complaint"
    assert detect_high_risk("Хочу пожаловаться на участника") == "complaint"


def test_detect_high_risk_no_false_positives():
    assert detect_high_risk("прошу вернуть мне доступ к школе") is None
    assert detect_high_risk("примите меры чтобы открыть запись") is None
    assert detect_high_risk("я хочу возврат в поток") is None
    assert detect_high_risk("прошу вернуть код") is None
    assert detect_high_risk("Не работает приложение") is None
    assert detect_high_risk("Хочу стать учеником на наставничестве") is None


def test_high_risk_reply_texts_are_structured():
    refund = high_risk_reply("refund")
    assert "ФИО" in refund
    assert "скриншот" in refund.lower() or "чек" in refund.lower()

    pdn = high_risk_reply("personal_data")
    assert "удалить" in pdn.lower()
    assert "email" in pdn.lower() or "телефон" in pdn.lower()

    complaint = high_risk_reply("complaint")
    assert "произошло" in complaint.lower() or "обращение" in complaint.lower()


def test_pipeline_high_risk_escalates_with_collection_text():
    class HighRiskKB:
        def match_many(self, q):
            return [
                MatchResult(
                    MatchMode.NO_MATCH,
                    1.0,
                    None,
                    None,
                    None,
                    [],
                    decision_reason="high_risk:refund",
                    subquestion=q,
                )
            ]

        def match(self, q):
            return self.match_many(q)[0]

    pipeline = DecisionPipeline(HighRiskKB(), _FakeLLM())
    msg = InboundMessage(
        "1", "k", "t", "c", Channel.CONVERSATION,
        "Прошу вернуть денежные средства",
    )
    result = asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.traces[0].escalation_reason == EscalationReason.HIGH_RISK_REFUND
    assert result.actions
    assert result.actions[0].escalate is True
    assert "ФИО" in (result.actions[0].text or "")


def test_top3_is_user_visible_when_enabled(monkeypatch):
    from app.application import decision_pipeline as module

    monkeypatch.setattr(module.settings, "USER_VISIBLE_TOP3", True)

    class DummyKB:
        def get_category_choice_candidates(self, q, top_n=3):
            return []

    pipeline = DecisionPipeline(DummyKB(), _FakeLLM())
    match = MatchResult(
        MatchMode.CATEGORY_CHOICE,
        0.8,
        "Cat A",
        None,
        None,
        [],
        choice_candidates=[
            {"kb_id": "a", "choice_label": "Вариант A", "template_answer": "Ответ A"},
            {"kb_id": "b", "choice_label": "Вариант B", "template_answer": "Ответ B"},
        ],
        decision_reason="ambiguous",
        subquestion="q",
    )
    trace, actions = asyncio.run(pipeline.decide_match(match, "q"))
    assert trace.decision == Decision.CHOICE
    assert len(actions) == 1
    assert "Вариант A" in (actions[0].text or "")
    assert "Вариант B" in (actions[0].text or "")


def test_inbound_query_is_logged(caplog):
    class QuietKB:
        def match_many(self, q):
            return [
                MatchResult(
                    MatchMode.NO_MATCH,
                    0.0,
                    None,
                    None,
                    None,
                    [],
                    decision_reason="noise:meta_noise",
                )
            ]

        def match(self, q):
            return self.match_many(q)[0]

    pipeline = DecisionPipeline(QuietKB(), _FakeLLM())
    msg = InboundMessage(
        "evt-1", "key", "topic", "entity-9", Channel.CONVERSATION,
        "Бесполезное сообщение",
    )
    with caplog.at_level(logging.INFO, logger="decision_pipeline"):
        asyncio.run(pipeline.decide(msg))
    joined = " ".join(r.message for r in caplog.records)
    assert "inbound_query" in joined
    assert "entity-9" in joined
    assert "Бесполезное сообщение" in joined


def test_choice_state_operator_request_escalates_directly():
    import asyncio
    from app.application.support_service import SupportApplication
    from app.domain.enums import Channel, ConversationStateKind, EscalationReason
    from app.domain.models import InboundMessage
    from app.infrastructure.persistence.models import InboundEvent

    class DummyPipeline:
        def decide(self, *_args, **_kwargs):
            raise AssertionError("classifier should not be called for explicit operator request")
    class DummyIntercom: pass
    app = SupportApplication(DummyPipeline(), DummyIntercom())

    class DummyState:
        state = ConversationStateKind.CHOOSING
        data = {"candidate_ids": ["a", "b"], "candidates": [], "attempts": 0}

    called = {}
    async def fake_escalate(session, message, event, reason, reply_text=None):
        called["reason"] = reason
    app._escalate = fake_escalate

    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "оператор")
    event = InboundEvent(id="ev", event_id="ev") if False else object()
    result = asyncio.run(app._handle_state(None, event, message, DummyState()))
    assert result is True
    assert called["reason"] == EscalationReason.USER_REQUESTED_OPERATOR


def test_real_world_missed_event_code_question_routes_to_kb_003():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Как получить код, если меня не будет на мероприятии?") == "kb_003"


def test_real_world_telegram_binding_failure_routes_to_app_failure():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "Не могу привязать свой Телеграмм аккаунт. По кнопке не переходит. Не работает в приложении"
    ) == "kb_001"


def test_real_world_account_lost_after_update_routes_to_app_failure():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "После обновления пропал мой аккаунт, считывает как новый без данных в приложении"
    ) == "kb_001"


def test_real_world_verification_call_failure_routes_to_phone_verification():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Позвонил на указанный номер но не вошел") == "kb_027"


def test_combined_login_and_registration_failure_prefers_app_failure():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Не могу войти и зарегистрироваться на следующий поток") == "kb_001"


def test_registration_with_missing_telegram_is_not_misclassified_as_registration_failure():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "При регистрации на запись в школу требуется заполнить поле Telegram, но у меня нет/не работает Telegram"
    ) != "kb_002"


def test_inbound_and_routing_logs_keep_full_text(caplog):
    class QuietKB:
        def match_many(self, q):
            return [
                MatchResult(
                    MatchMode.NO_MATCH, 0.0, None, None, None, [],
                    decision_reason="reason-" + "x" * 300, subquestion=q,
                )
            ]
        def match(self, q):
            return self.match_many(q)[0]

    pipeline = DecisionPipeline(QuietKB(), _FakeLLM())
    text = "Полный текст обращения " + ("деталь " * 80)
    msg = InboundMessage("evt-full", "key", "topic", "entity-full", Channel.CONVERSATION, text)
    with caplog.at_level(logging.INFO, logger="decision_pipeline"):
        asyncio.run(pipeline.decide(msg))
    joined = " ".join(r.message for r in caplog.records)
    assert text in joined
    assert "reason-" + "x" * 300 in joined


def test_online_training_hard_rule_routes_to_dedicated_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "Здравствуйте! Какие есть онлайн-школы и как записаться на онлайн обучение?"
    ) == "kb_047"


def test_online_training_answer_is_exact_and_routable():
    data = json.loads(Path("data/knowledge_base.json").read_text())
    entry = next(item for item in data if item.get("id") == "kb_047")
    assert entry["template_answer"] == (
        "Список актуальных онлайн-школ опубликован в данном Telegram-канале. https://t.me/terraonlinescho\n\n"
        "Пожалуйста, ознакомьтесь с доступными направлениями и выберите интересующую вас школу. "
        "Все инструкции по записи также размещены в канале."
    )


def test_online_mentorship_stays_separate_from_online_training():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "Есть ли онлайн наставничество? Хочу работать с наставником удаленно."
    ) == "kb_040"


def test_online_training_is_not_triggered_by_generic_online_word():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Есть ли онлайн формат мероприятия 9 сентября?") != "kb_047"


def test_online_training_handles_common_russian_variants():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    for question in (
        "Подскажите, есть ли дистанционное обучение?",
        "Хочу учиться онлайн, какие школы доступны?",
        "Где посмотреть список онлайн школ?",
    ):
        assert kb._hard_rule_match(question) == "kb_047"
