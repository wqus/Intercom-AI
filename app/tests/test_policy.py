import asyncio
from app.application.decision_pipeline import DecisionPipeline
from app.domain.models import InboundMessage
from app.domain.enums import EscalationReason
from app.domain.enums import Channel, Decision
from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult

class FakeKB:
    def match_many(self, q):
        return [MatchResult(MatchMode.TEMPLATE, .95, "cat", "answer", "kb_1", [], decision_reason="test")]
    def match(self,q): return self.match_many(q)[0]

class FakeLLM: pass

def test_template_decision():
    pipeline=DecisionPipeline(FakeKB(), FakeLLM())
    msg=InboundMessage("1","k","t","c",Channel.CONVERSATION,"нужна помощь по обучению")
    result=asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision==Decision.ANSWER_TEMPLATE
    assert result.actions[0].text.startswith("Добрый день!\n\nanswer")
    assert result.actions[0].text.count("Добрый день!") == 1
    assert result.actions[0].text.count("Если этот ответ не помог") == 1



class NoiseKB:
    def match_many(self, q):
        return [MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], decision_reason="noise:meta_noise")]
    def match(self, q): return self.match_many(q)[0]

def test_noise_is_ignored_without_reply_or_escalation():
    pipeline = DecisionPipeline(NoiseKB(), FakeLLM())
    msg = InboundMessage("1","k","t","c",Channel.CONVERSATION,"Бесполезное сообщение")
    result = asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision == Decision.IGNORE
    assert result.actions == []


def test_meta_noise_produces_no_reply():
    from app.application.decision_pipeline import DecisionPipeline
    from app.domain.models import InboundMessage
    from app.domain.enums import Channel
    from app.infrastructure.ai.knowledge_base import MatchResult, MatchMode

    class DummyKB:
        def match_many(self, _text):
            return [MatchResult(MatchMode.NO_MATCH, 0.0, None, None, None, [], decision_reason="noise:meta_noise")]

    class DummyLLM:
        async def classify(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called for noise")

    pipeline = DecisionPipeline(DummyKB(), DummyLLM())
    msg = InboundMessage(
        event_id="e1", event_key="k1", topic="t", entity_id="c1",
        channel=Channel.CONVERSATION, text="Бесполезное сообщение",
    )

    import asyncio
    result = asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision.value == "ignore"
    assert result.actions == []


def test_category_choice_is_user_visible_and_not_llm(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", True)

    class FakeKB:
        entries_by_id = {}
    class FakeLLM:
        async def classify(self, *_args, **_kwargs):
            raise AssertionError("Top-3 choice must not call LLM")

    pipeline = DecisionPipeline(FakeKB(), FakeLLM())
    from app.infrastructure.ai.knowledge_base import MatchResult, MatchMode
    match = MatchResult(
        MatchMode.CATEGORY_CHOICE, 0.8, "Cat A", None, "kb_1", [],
        choice_candidates=[
            {"kb_id":"kb_1","choice_label":"A","template_answer":"A answer"},
            {"kb_id":"kb_2","choice_label":"B","template_answer":"B answer"},
        ],
        decision_reason="ambiguous", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.CHOICE
    assert trace.llm_used is False
    assert trace.choice_candidate_ids == ["kb_1", "kb_2"]
    assert actions and "1. A" in (actions[0].text or "") and "2. B" in (actions[0].text or "")

def test_llm_mode_selects_and_validates_template(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", True)

    class FakeKB:
        entries_by_id = {
            "kb_1": {"id": "kb_1", "category": "A", "template_answer": "Answer A"},
        }
        def validate_llm_selection(self, q, kb_id):
            return True, "ok", {}

    class FakeLLM:
        async def classify(self, question, candidates):
            from app.infrastructure.ai.llm_client import LLMResult
            assert candidates
            return LLMResult(["kb_1"])

    pipeline = DecisionPipeline(FakeKB(), FakeLLM())
    from app.infrastructure.ai.knowledge_base import MatchResult, MatchMode
    match = MatchResult(
        MatchMode.LLM, 0.6, "Cat A", None, "kb_1", [{"id": "kb_1"}],
        decision_reason="middle_band", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.ANSWER_LLM
    assert trace.llm_used is True
    assert actions and actions[0].kind == "business_answer"
    assert actions[0].text == "Answer A"


def test_llm_escalation_is_the_only_fallback(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", True)

    class FakeKB:
        entries_by_id = {}
        def validate_llm_selection(self, q, kb_id):
            return False, "bad", {}

    class FakeLLM:
        async def classify(self, *_args, **_kwargs):
            from app.infrastructure.ai.llm_client import LLMResult
            return LLMResult([], escalation=True, reason="model_escalation")

    pipeline = DecisionPipeline(FakeKB(), FakeLLM())
    match = MatchResult(
        MatchMode.LLM, 0.6, "Cat A", None, "kb_1",
        [{"id": "kb_1"}], decision_reason="supported_intent_needs_llm", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.ESCALATE
    assert trace.llm_used is True
    assert trace.escalation_reason == EscalationReason.LLM_CANNOT_ANSWER
    assert actions and actions[0].escalate is True


def test_trace_preserves_all_quality_metrics():
    from app.application.decision_pipeline import DecisionPipeline
    from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult

    match = MatchResult(
        MatchMode.TEMPLATE, 0.7742, "Cat", "answer", "kb_1", [],
        margin=0.0421,
        embedding_score=0.9123,
        semantic_score=0.7345,
        lexical_score=0.8123,
        fuzzy_score=0.9678,
        intent_fit_score=0.7012,
        family_fit_score=0.6432,
        evidence_score=0.5521,
        real_intent_evidence=0.4988,
        query_quality_score=0.4123,
        domain_support_score=0.3777,
        ontology_alignment_score=0.6888,
        negative_contrast_score=0.0911,
        contrastive_margin=0.1337,
        contrastive_competitor_id="kb_2",
        contrastive_reason="positive_vs_negative",
        decision_reason="test_metrics",
    )

    class FakeKB: pass
    trace, _ = __import__('asyncio').run(
        DecisionPipeline(FakeKB(), object()).decide_match(match, "q")
    )

    assert trace.score == 0.7742
    assert trace.margin == 0.0421
    assert trace.embedding_score == 0.9123
    assert trace.semantic_score == 0.7345
    assert trace.lexical_score == 0.8123
    assert trace.fuzzy_score == 0.9678
    assert trace.intent_fit_score == 0.7012
    assert trace.family_fit_score == 0.6432
    assert trace.evidence_score == 0.5521
    assert trace.real_intent_evidence == 0.4988
    assert trace.query_quality_score == 0.4123
    assert trace.domain_support_score == 0.3777
    assert trace.ontology_alignment_score == 0.6888
    assert trace.negative_contrast_score == 0.0911
    assert trace.contrastive_margin == 0.1337
    assert trace.contrastive_competitor_id == "kb_2"
    assert trace.contrastive_reason == "positive_vs_negative"


def test_oos_decision_escalates_for_substantive_message():
    import asyncio
    from app.application.decision_pipeline import DecisionPipeline
    from app.domain.enums import Decision
    from app.domain.models import InboundMessage
    from app.domain.enums import Channel

    class FakeKB:
        def match_many(self, text):
            from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult
            return [MatchResult(MatchMode.NO_MATCH, 0.42, None, None, None, [], subquestion=text, decision_reason="oos:no_supported_intent")]

    pipeline = DecisionPipeline(FakeKB(), object())
    result = asyncio.run(pipeline.decide(InboundMessage("e","k","t","c",Channel.CONVERSATION,"Как оформить налоговый вычет?")))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.traces[0].escalation_reason == EscalationReason.NO_CATEGORY
    assert result.actions and result.actions[0].escalate is True
    assert "передан оператору" in result.actions[0].text.lower()


def test_standalone_one_word_is_ignored_but_operator_is_not():
    import asyncio
    from app.application.decision_pipeline import DecisionPipeline
    from app.domain.enums import Decision, Channel, EscalationReason
    from app.domain.models import InboundMessage

    class FakeKB:
        def match_many(self, text):
            raise AssertionError("KB must not be reached")

    pipeline = DecisionPipeline(FakeKB(), object())
    one_word = asyncio.run(pipeline.decide(InboundMessage("e","k","t","c",Channel.CONVERSATION,"Телеграм")))
    assert one_word.traces[0].decision == Decision.IGNORE
    assert one_word.actions == []

    operator = asyncio.run(pipeline.decide(InboundMessage("e2","k2","t","c2",Channel.CONVERSATION,"оператор")))
    assert operator.traces[0].decision == Decision.ESCALATE
    assert operator.actions and operator.actions[0].escalate
