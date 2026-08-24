import asyncio
from app.application.decision_pipeline import DecisionPipeline
from app.domain.models import InboundMessage
from app.domain.enums import Channel, Decision
from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult

class FakeKB:
    def match_many(self, q):
        return [MatchResult(MatchMode.TEMPLATE, .95, "cat", "answer", "kb_1", [], decision_reason="test")]
    def match(self,q): return self.match_many(q)[0]

class FakeLLM: pass

def test_template_decision():
    pipeline=DecisionPipeline(FakeKB(), FakeLLM())
    msg=InboundMessage("1","k","t","c",Channel.CONVERSATION,"hello")
    result=asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision==Decision.ANSWER_TEMPLATE
    assert result.actions[0].text=="answer"



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


def test_top3_output_is_disabled_by_default(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "USER_VISIBLE_TOP3", False)

    class FakeKB:
        def get_category_choice_candidates(self, q): return []
    class FakeLLM: pass
    from app.infrastructure.ai.knowledge_base import MatchResult, MatchMode
    pipeline = DecisionPipeline(FakeKB(), FakeLLM())
    match = MatchResult(
        MatchMode.CATEGORY_CHOICE, 0.8, "Cat A", None, None, [],
        choice_candidates=[{"kb_id":"a","choice_label":"A"},{"kb_id":"b","choice_label":"B"}],
        decision_reason="ambiguous", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision.value == 'choice'
    assert actions == []


def test_llm_route_never_calls_llm_even_if_config_is_misconfigured(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", True)
    monkeypatch.setattr(module.settings, "USER_VISIBLE_TOP3", True)

    class FakeKB:
        def get_category_choice_candidates(self, q, top_n=3):
            return [
                {"kb_id":"kb_1", "choice_label":"Вариант 1"},
                {"kb_id":"kb_2", "choice_label":"Вариант 2"},
                {"kb_id":"kb_3", "choice_label":"Вариант 3"},
            ]

    class NeverCallLLM:
        async def classify(self, *_args, **_kwargs):
            raise AssertionError("LLM must never be called in conservative mode")

    pipeline = DecisionPipeline(FakeKB(), NeverCallLLM())
    match = MatchResult(
        MatchMode.LLM, 0.8, "Cat A", None, "kb_1",
        [{"kb_id":"kb_1","category":"Cat A"}],
        decision_reason="supported_intent_needs_llm", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.CHOICE
    assert trace.llm_used is False
    assert len(actions) == 1


def test_llm_route_is_converted_to_top3_without_calling_llm(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", False)
    monkeypatch.setattr(module.settings, "USER_VISIBLE_TOP3", True)

    class FakeKB:
        def get_category_choice_candidates(self, q, top_n=3):
            return [
                {"kb_id":"kb_1", "choice_label":"Вариант 1"},
                {"kb_id":"kb_2", "choice_label":"Вариант 2"},
                {"kb_id":"kb_3", "choice_label":"Вариант 3"},
            ]

    class NeverCallLLM:
        async def classify(self, *_args, **_kwargs):
            raise AssertionError("LLM must never be called in conservative mode")

    pipeline = DecisionPipeline(FakeKB(), NeverCallLLM())
    match = MatchResult(
        MatchMode.LLM, 0.80, "Cat A", None, "kb_1",
        [{"kb_id":"kb_1","category":"Cat A"}],
        decision_reason="supported_intent_needs_llm", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.CHOICE
    assert trace.llm_used is False
    assert len(actions) == 1
    assert "1. Вариант 1" in actions[0].text
    assert "2. Вариант 2" in actions[0].text
    assert "3. Вариант 3" in actions[0].text


def test_llm_route_falls_back_to_rephrase_when_no_safe_top3(monkeypatch):
    from app.application import decision_pipeline as module
    monkeypatch.setattr(module.settings, "LLM_ENABLED", False)

    class FakeKB:
        def get_category_choice_candidates(self, q, top_n=3):
            return []

    class NeverCallLLM:
        async def classify(self, *_args, **_kwargs):
            raise AssertionError("LLM must never be called in conservative mode")

    pipeline = DecisionPipeline(FakeKB(), NeverCallLLM())
    match = MatchResult(
        MatchMode.LLM, 0.60, "Cat A", None, "kb_1",
        [{"kb_id":"kb_1","category":"Cat A"}],
        decision_reason="supported_intent_needs_llm", subquestion="q"
    )
    trace, actions = __import__('asyncio').run(pipeline.decide_match(match, 'q'))
    assert trace.decision == Decision.REPHRASE
    assert trace.llm_used is False
    assert actions and "переформулируйте" in actions[0].text


def test_escalation_delivery_secondary_failure_does_not_bubble():
    import asyncio
    from app.application.support_service import SupportApplication
    from app.domain.enums import Channel
    from app.domain.models import InboundMessage

    class FakePipeline: pass

    class FakeIntercom:
        async def send_reply(self, *args): return True
        async def add_escalation_tag(self, *args): raise RuntimeError("tag failed")

    class FakeTelegram:
        async def notify_escalation(self, **kwargs): raise RuntimeError("telegram failed")

    app = SupportApplication(FakePipeline(), FakeIntercom(), FakeTelegram())
    msg = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "оператор")
    asyncio.run(app._deliver(msg, "Передаю оператору", True))




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
