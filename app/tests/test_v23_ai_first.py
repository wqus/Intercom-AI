import asyncio

from app.application.decision_pipeline import DecisionPipeline
from app.domain.enums import Channel, Decision
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction
from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult, is_non_query_message
from app.infrastructure.ai.llm_client import LLMResult


def _match(entries):
    return MatchResult(
        MatchMode.LLM, 0.31, "cat", None, entries[0]["id"], entries,
        subquestion="q", decision_reason="ai_first_semantic_judge",
        semantic_score=0.45, intent_fit_score=0.20, domain_support_score=0.06,
    )


def test_short_domain_words_are_not_noise():
    assert not is_non_query_message("Оплата")
    assert not is_non_query_message("Регистрация")
    assert not is_non_query_message("Телеграм")
    assert is_non_query_message("Спасибо")


def test_first_acknowledgement_is_not_ignored():
    class KB:
        def match_many(self, text):
            return [MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], subquestion=text, decision_reason="no_defensible_retrieval_neighborhood")]
    result = asyncio.run(DecisionPipeline(KB(), None).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "спасибо"),
        is_first_message=True,
    ))
    assert result.traces[0].decision == Decision.ESCALATE


def test_llm_choice_response_is_invalid_and_uses_top3_only_as_error_fallback():
    class KB:
        entries_by_id = {
            "a": {"id": "a", "category": "A", "template_answer": "A"},
            "b": {"id": "b", "category": "B", "template_answer": "B"},
            "c": {"id": "c", "category": "C", "template_answer": "C"},
        }
    class LLM:
        async def classify(self, q, candidates, context=None):
            return LLMResult(["a", "b"], decision="choice")
    match = MatchResult(
        MatchMode.LLM, 0.7, "cat", None, "a",
        [{"id":"a"},{"id":"b"},{"id":"c"}],
        top_candidates=[
            {"id":"a","choice_label":"A"},
            {"id":"b","choice_label":"B"},
            {"id":"c","choice_label":"C"},
        ],
        subquestion="q", decision_reason="ai_first_semantic_judge",
    )
    trace, actions = asyncio.run(DecisionPipeline(KB(), LLM()).decide_match(match, "q"))
    assert trace.decision == Decision.CHOICE
    assert trace.llm_used is True
    assert trace.decision_reason.endswith("llm_error_top3_fallback")
    assert trace.choice_candidate_ids == ["a", "b", "c"]


def test_short_domain_message_reaches_llm_after_retrieval():
    captured = {}
    class KB:
        entries_by_id = {"a": {"id": "a", "category": "Оплата", "template_answer": "A"}}
        def match_many(self, text):
            captured["question"] = text
            return [MatchResult(
                MatchMode.LLM, 0.4, "Оплата", None, "a", [{"id": "a"}],
                top_candidates=[{"id": "a", "choice_label": "Оплата"}],
                subquestion=text, decision_reason="ai_first_semantic_judge",
            )]
    class LLM:
        async def classify(self, question, candidates, context=None):
            captured["llm_question"] = question
            captured["candidates"] = candidates
            return LLMResult(["a"])

    result = asyncio.run(DecisionPipeline(KB(), LLM()).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "оплата"),
        is_first_message=False,
    ))
    assert captured["question"] == "оплата"
    assert captured["llm_question"] == "оплата"
    assert captured["candidates"] == [{"id": "a"}]
    assert result.traces[0].decision == Decision.ANSWER_LLM

def test_ai_receives_conversation_context():
    captured = {}
    class KB:
        entries_by_id = {"a": {"id": "a", "category": "A", "template_answer": "A"}}
        def validate_llm_selection(self, q, kb_id, evidence=None): return True, "ok", {}
    class LLM:
        async def classify(self, q, candidates, context=None):
            captured["context"] = context
            return LLMResult(["a"])
    asyncio.run(DecisionPipeline(KB(), LLM()).decide_match(
        _match([{"id": "a"}]), "q", [{"role":"user","text":"не могу войти"}]
    ))
    assert captured["context"] == [{"role":"user","text":"не могу войти"}]


def test_non_noise_no_match_always_gets_user_visible_fallback():
    class KB:
        def match_many(self, text):
            return [MatchResult(
                MatchMode.NO_MATCH, 0.0, None, None, None, [],
                subquestion=text, decision_reason="no_defensible_retrieval_neighborhood",
            )]
    result = asyncio.run(DecisionPipeline(KB(), None).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "это содержательный вопрос"),
        is_first_message=False,
    ))
    assert result.actions
    assert any(action.kind == "reply" and action.text for action in result.actions)
    assert result.traces[0].decision == Decision.ESCALATE


def test_acknowledgement_is_silent_only_on_later_turn():
    class KB:
        def match_many(self, text):
            raise AssertionError("acknowledgement should be filtered before classification")
    pipeline = DecisionPipeline(KB(), None)
    result = asyncio.run(pipeline.decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "спасибо"),
        is_first_message=False,
    ))
    assert result.traces[0].decision == Decision.IGNORE
    assert result.actions == []


def test_first_acknowledgement_is_not_silent():
    class KB:
        def match_many(self, text):
            return [MatchResult(
                MatchMode.NO_MATCH, 0.0, None, None, None, [],
                subquestion=text, decision_reason="noise:meta_noise",
            )]
    result = asyncio.run(DecisionPipeline(KB(), None).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "спасибо"),
        is_first_message=True,
    ))
    assert result.actions
    assert result.traces[0].decision == Decision.ESCALATE
    # The first-turn policy must not silently drop the event.
    assert any(action.text for action in result.actions)


def test_llm_choice_requires_two_or_three_unique_ids():
    class KB:
        entries_by_id = {"a": {"id": "a", "category": "A", "template_answer": "A"}}
        def validate_llm_selection(self, q, kb_id, evidence=None): return True, "ok", {}
    class LLM:
        async def classify(self, q, candidates, context=None):
            return LLMResult(["a"], decision="choice")
    trace, actions = asyncio.run(DecisionPipeline(KB(), LLM()).decide_match(
        _match([{"id": "a"}, {"id": "b"}]), "q"
    ))
    assert trace.decision == Decision.ESCALATE
    assert actions and actions[0].escalate


def test_llm_clarify_requires_exactly_one_id():
    class KB:
        entries_by_id = {
            "a": {"id": "a", "category": "A", "template_answer": "A", "clarification_question": "Q"},
            "b": {"id": "b", "category": "B", "template_answer": "B", "clarification_question": "Q"},
        }
        def validate_llm_selection(self, q, kb_id, evidence=None): return True, "ok", {}
    class LLM:
        async def classify(self, q, candidates, context=None):
            return LLMResult(["a", "b"], decision="clarify")
    trace, actions = asyncio.run(DecisionPipeline(KB(), LLM()).decide_match(
        _match([{"id": "a"}, {"id": "b"}]), "q"
    ))
    assert trace.decision == Decision.ESCALATE
    assert actions and actions[0].escalate


def test_llm_answer_multiple_ids_is_allowed_only_as_explicit_multi_intent_contract():
    class KB:
        entries_by_id = {
            "a": {"id": "a", "category": "A", "template_answer": "A"},
            "b": {"id": "b", "category": "B", "template_answer": "B"},
        }
        def validate_llm_selection(self, q, kb_id, evidence=None): return True, "ok", {}
    class LLM:
        async def classify(self, q, candidates, context=None):
            return LLMResult(["a", "b"], decision="answer")
    trace, actions = asyncio.run(DecisionPipeline(KB(), LLM()).decide_match(
        _match([{"id": "a"}, {"id": "b"}]), "q"
    ))
    assert trace.decision == Decision.ANSWER_LLM
    assert len(actions) == 2
    assert {a.text for a in actions} == {"A", "B"}


def test_kb_has_no_duplicate_positive_examples_across_intents():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json").read_text(encoding="utf-8"))
    owners = {}
    for entry in data:
        if entry.get("id") == "default":
            continue
        for phrase in entry.get("positive_examples", []):
            owners.setdefault(phrase.strip().lower().replace("ё", "е"), set()).add(entry["id"])
    collisions = {phrase: ids for phrase, ids in owners.items() if len(ids) > 1}
    assert collisions == {}


def test_invalid_llm_decision_fails_closed_to_escalation():
    result = LLMResult(["a"], decision="unexpected")
    assert result.decision == "escalate"
    assert result.escalation is True


def test_contradictory_llm_escalation_flag_wins():
    result = LLMResult(["a"], escalation=True, decision="answer")
    assert result.decision == "escalate"
    assert result.escalation is True


def test_first_greeting_is_escalated_not_clarified_or_ignored():
    class KB:
        def match_many(self, text):
            raise AssertionError("first-turn greeting must be handled before KB")
    result = asyncio.run(DecisionPipeline(KB(), None).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "привет"),
        is_first_message=True,
    ))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.actions and result.actions[0].escalate


def test_later_greeting_is_silent_contextual_noise():
    class KB:
        def match_many(self, text):
            raise AssertionError("later greeting must be filtered before KB")
    result = asyncio.run(DecisionPipeline(KB(), None).decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "привет"),
        is_first_message=False,
    ))
    assert result.traces[0].decision == Decision.IGNORE
    assert result.actions == []
