"""v22 routing regressions: high-risk, Top-3 visibility, inbound logging path."""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.application.decision_pipeline import DecisionPipeline
from app.domain.enums import Channel, Decision, EscalationReason
from app.domain.models import InboundMessage
from app.domain.policy import high_risk_reply
from app.infrastructure.ai.knowledge_base import (
    MatchMode,
    MatchResult,
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
