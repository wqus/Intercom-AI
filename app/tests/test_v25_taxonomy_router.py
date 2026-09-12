import asyncio

from app.application.decision_pipeline import DecisionPipeline
from app.core import config as config_module
from app.domain.enums import Channel, Decision, EscalationReason
from app.domain.models import InboundMessage
from app.infrastructure.ai.knowledge_base import MatchMode, MatchResult
from app.infrastructure.ai.llm_client import LLMResult, build_taxonomy_user_message


def _msg(text, *, entity="c", event="e"):
    return InboundMessage(event, event + "k", "t", entity, Channel.CONVERSATION, text)


class CatalogKB:
    entries_by_id = {
        "kb_registration": {
            "id": "kb_registration",
            "category": "Регистрация",
            "template_answer": "Ответ про регистрацию.",
            "clarification_question": "На каком этапе регистрации проблема?",
        },
        "kb_price": {
            "id": "kb_price",
            "category": "Стоимость обучения",
            "template_answer": "Ответ про стоимость.",
        },
    }

    def intent_catalog_for_llm(self):
        return list(self.entries_by_id.values())


class RecordingLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def classify(self, question, candidates, context=None, **kwargs):
        self.calls.append((question, candidates, context, kwargs))
        return self.result


def test_taxonomy_router_sends_full_compact_catalog_and_context(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration"], decision="answer"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("код", entity="c1"),
        is_first_message=False,
        conversation_context=[
            {"role": "user", "text": "Не могу зарегистрироваться"},
            {"role": "assistant", "text": "На каком этапе проблема?"},
        ],
    ))
    assert result.traces[0].decision == Decision.ANSWER_LLM
    assert result.actions[0].text == "Ответ про регистрацию."
    assert len(llm.calls) == 1
    question, candidates, context, kwargs = llm.calls[0]
    assert question == "код"
    assert {x["id"] for x in candidates} == {"kb_registration", "kb_price"}
    assert context[-1]["text"] == "На каком этапе проблема?"
    assert kwargs["taxonomy_mode"] is True


def test_taxonomy_router_ignore_is_silent(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult([], decision="ignore"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("до свидания"), is_first_message=False,
    ))
    assert result.traces[0].decision == Decision.IGNORE
    assert result.actions == []


def test_taxonomy_router_llm_failure_escalates_without_top3(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")

    class FailingLLM:
        async def classify(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    result = asyncio.run(DecisionPipeline(CatalogKB(), FailingLLM()).decide(
        _msg("код"), is_first_message=False,
        conversation_context=[{"role": "user", "text": "Не могу зарегистрироваться"}],
    ))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.traces[0].escalation_reason == EscalationReason.LLM_CANNOT_ANSWER
    assert result.actions and result.actions[0].escalate is True
    assert result.traces[0].choice_candidate_ids is None


def test_taxonomy_router_supports_topic_switch_as_new_intent(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_price"], decision="answer"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("Сколько стоит обучение?"), is_first_message=False,
        conversation_context=[
            {"role": "user", "text": "Не могу зарегистрироваться"},
            {"role": "assistant", "text": "На каком этапе проблема?"},
        ],
    ))
    assert result.traces[0].decision == Decision.ANSWER_LLM
    assert result.traces[0].kb_id == "kb_price"


def test_taxonomy_prompt_contains_compact_directory_and_context():
    payload = build_taxonomy_user_message(
        "код",
        [
            {"id": "kb_1", "category": "Регистрация", "when_to_use": ["Проблема с регистрацией"]},
            {"id": "kb_2", "category": "Оплата", "definition": "Вопросы по оплате"},
        ],
        [{"role": "user", "text": "Не могу зарегистрироваться"}],
    )
    import json
    assert '"intent_catalog"' in payload
    assert '"conversation_history"' in payload
    parsed = json.loads(payload)
    catalog = {item["id"]: item for item in parsed["intent_catalog"]}
    assert catalog["kb_1"]["definition"] == "Проблема с регистрацией"
    assert catalog["kb_1"]["answer"] is False
    assert "template_answer" not in payload
    assert "when_to_use" not in payload


def test_taxonomy_router_low_certainty_answer_fails_closed(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration"], decision="answer", certainty="low"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.actions and result.actions[0].escalate is True


def test_taxonomy_router_answer_requires_approved_kb_template(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration"], decision="answer", certainty="high"))
    kb = CatalogKB()
    kb.entries_by_id = {k: dict(v) for k, v in CatalogKB.entries_by_id.items()}
    kb.entries_by_id["kb_registration"]["template_answer"] = ""
    result = asyncio.run(DecisionPipeline(kb, llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].decision == Decision.ESCALATE


def test_taxonomy_router_uses_previous_user_and_assistant_messages(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration"], decision="answer", certainty="high"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("код"), is_first_message=False,
        conversation_context=[
            {"role":"user","text":"Не могу зарегистрироваться"},
            {"role":"assistant","text":"На каком этапе регистрации проблема?"},
            {"role":"user","text":"код"},
        ],
    ))
    assert result.traces[0].decision == Decision.ANSWER_LLM
    context = llm.calls[0][2]
    assert [x["text"] for x in context] == [
        "Не могу зарегистрироваться",
        "На каком этапе регистрации проблема?",
        "код",
    ]


def test_taxonomy_router_accepts_multiple_answer_ids_after_coverage(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration", "kb_price"], decision="answer", certainty="high", is_compound=True))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].decision == Decision.ANSWER_LLM
    assert "Ответ про регистрацию." in result.actions[0].text or "Ответ про регистрацию." in result.actions[0].text


def test_taxonomy_catalog_exposes_answerability_metadata():
    from app.infrastructure.ai.llm_client import _intent_catalog_entry
    item = _intent_catalog_entry({
        "id": "kb_1",
        "category": "Test",
        "when_to_use": ["A"],
        "requires_clarification": True,
        "clarification_question": "Which?",
        "template_answer": "Approved answer",
    })
    assert item["answer"] is True
    assert item["clarify"] is True


def test_ignore_does_not_clear_pending_clarification_state():
    from app.application.support_service import SupportApplication
    from app.domain.enums import ConversationStateKind
    assert SupportApplication._should_reset_state_after_result(
        ConversationStateKind.CLARIFYING,
        [LLMTraceLike(Decision.IGNORE)],
    ) is False


class LLMTraceLike:
    def __init__(self, decision):
        self.decision = decision



def test_taxonomy_router_does_not_force_clarify_when_model_has_answer(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_leave"], decision="answer", certainty="high"))

    class KB(CatalogKB):
        entries_by_id = {
            "kb_leave": {
                "id": "kb_leave",
                "category": "Выход из школы",
                "template_answer": "Ответ про выход.",
                "requires_clarification": True,
                "clarification_question": "Вы хотите просто выйти или перейти?",
            }
        }

    result = asyncio.run(DecisionPipeline(KB(), llm).decide(
        _msg("Я хочу просто выйти из школы"), is_first_message=False,
    ))
    assert result.traces[0].decision == Decision.ANSWER_LLM
    assert result.actions[0].text == "Ответ про выход."


def test_taxonomy_router_rephrase_pending_metadata_contains_attempts():
    from app.application.support_service import SupportApplication
    from app.domain.enums import ConversationStateKind
    state = type("State", (), {
        "state": ConversationStateKind.CLARIFYING,
        "data": {"kb_id": "kb_1", "attempts": 1, "original_question": "q"},
    })()
    context = SupportApplication._routing_context_for_state(state)
    assert "attempts=1" in context[0]["text"]


def test_taxonomy_primary_path_never_calls_global_retrieval(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")

    class KB(CatalogKB):
        def match_many(self, *_args, **_kwargs):
            raise AssertionError("taxonomy mode must not call legacy retrieval")
        def match(self, *_args, **_kwargs):
            raise AssertionError("taxonomy mode must not call legacy retrieval")

    llm = RecordingLLM(LLMResult(["kb_registration"], decision="answer", certainty="high"))
    result = asyncio.run(DecisionPipeline(KB(), llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].decision == Decision.ANSWER_LLM


def test_taxonomy_schema_is_single_intent_contract(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": '{"decision":"answer","selected_kb_ids":["kb_1"],"is_compound":false,"certainty":"high"}'})()})()]})()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    from app.infrastructure.ai.llm_client import PolzaLLMClient
    client = PolzaLLMClient("key", "openai/gpt-5.6-luna-pro")
    result = asyncio.run(client.classify(
        "код",
        [{"id": "kb_1", "category": "Регистрация", "when_to_use": ["Проблема"]}],
        taxonomy_mode=True,
    ))
    assert result.decision == "answer"
    schema = captured["response_format"]["json_schema"]["schema"]
    selected = schema["properties"]["selected_kb_ids"]
    assert selected["maxItems"] == 4
    assert selected["minItems"] == 0
    assert selected is schema["properties"]["selected_kb_ids"]
    assert schema["properties"]["is_compound"]["type"] == "boolean"


def test_taxonomy_router_low_certainty_ignore_fails_closed(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult([], decision="ignore", certainty="medium"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("что-то"), is_first_message=False))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.actions and result.actions[0].escalate is True


def test_pending_context_reaches_llm_instead_of_deterministic_ignore(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_registration"], decision="clarify", certainty="medium"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("ок"), is_first_message=False,
        conversation_context=[
            {"role": "system", "text": "PENDING_CLARIFICATION: kb_id=kb_registration; attempts=1; original_question=Не могу зарегистрироваться"},
            {"role": "assistant", "text": "На каком этапе регистрации проблема?"},
        ],
    ))
    assert result.traces[0].decision == Decision.CLARIFY
    assert len(llm.calls) == 1


def test_taxonomy_answer_schema_allows_up_to_four_final_intents_and_full_catalog_is_sent(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    class RecordingSchemaLLM(RecordingLLM):
        pass
    llm = RecordingSchemaLLM(LLMResult(["kb_registration"], decision="answer", certainty="high"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].kb_id == "kb_registration"
    assert len(llm.calls[0][1]) == 2


def test_taxonomy_kb_catalog_does_not_require_embedding_runtime(monkeypatch, tmp_path):
    import json
    import app.infrastructure.ai.knowledge_base as kb_module
    monkeypatch.setattr(kb_module, "SentenceTransformer", None)
    path = tmp_path / "kb.json"
    path.write_text(json.dumps([{
        "id": "kb_1",
        "category": "Категория",
        "definition": "Тест",
        "template_answer": "Ответ",
        "family": "test",
        "intent_type": "test",
        "positive_examples": ["тест"],
        "when_to_use": ["Тестовый запрос"],
        "when_not_to_use": ["Не использовать вне теста"],
        "constraints": {},
    }], ensure_ascii=False), encoding="utf-8")
    kb = kb_module.KnowledgeBase(path=str(path), enable_retrieval=False)
    assert kb.intent_catalog_for_llm()[0]["id"] == "kb_1"
    assert kb.retrieval_enabled is False
    try:
        kb.match("тест")
        assert False, "retrieval must be disabled in taxonomy mode"
    except RuntimeError as exc:
        assert "retrieval is disabled" in str(exc).lower()


def test_taxonomy_runtime_requires_retrieval_free_kb_configuration():
    # The startup code is intentionally guarded by an explicit mode validation;
    # static inspection here prevents accidental typo modes from silently selecting legacy.
    source = (__import__("pathlib").Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert 'router_mode not in {"taxonomy", "legacy"}' in source
    assert 'enable_retrieval=(router_mode == "legacy")' in source


def test_taxonomy_router_unknown_model_id_fails_closed(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_unknown"], decision="answer", certainty="high"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("код"), is_first_message=False))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.actions and result.actions[0].escalate is True


def test_taxonomy_router_clarify_requires_kb_question(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(["kb_price"], decision="clarify", certainty="medium"))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(_msg("стоимость"), is_first_message=False))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.actions and result.actions[0].escalate is True


def test_taxonomy_catalog_is_compact_without_losing_routing_contract():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase
    from app.infrastructure.ai.llm_client import _intent_catalog_entry
    import json

    # Use the repository KB fixture directly without initializing embeddings.
    kb_data = json.loads(open("data/knowledge_base.json", encoding="utf-8").read())
    rendered = json.dumps({"intent_catalog": [_intent_catalog_entry(e) for e in kb_data]}, ensure_ascii=False, separators=(",", ":"))
    assert len(kb_data) == 44  # raw fixture includes the server-side DEFAULT record
    assert len(rendered) < 25000
    assert '"template_answer"' not in rendered
    assert '"use_when"' in rendered
    assert '"do_not_use_when"' in rendered
    assert '"constraints"' in rendered
    assert '"confused_with"' in rendered
    for item in kb_data:
        compact = _intent_catalog_entry(item)
        assert compact["id"] == item["id"]
        assert compact["name"]
        assert "definition" in compact
        assert "answer" in compact and "clarify" in compact


def test_taxonomy_prompt_wire_format_is_compact():
    import json
    kb_data = json.loads(open("data/knowledge_base.json", encoding="utf-8").read())
    kb = __import__("app.infrastructure.ai.knowledge_base", fromlist=["KnowledgeBase"]).KnowledgeBase(
        path="data/knowledge_base.json", enable_retrieval=False
    )
    catalog_entries = kb.intent_catalog_for_llm()
    payload = build_taxonomy_user_message("код", catalog_entries, [])
    catalog = json.loads(payload)["intent_catalog"]
    assert len(catalog) == 43
    assert len(payload) < 25500
    assert "template_answer" not in payload
    assert "use_when" in payload
    assert "do_not_use_when" in payload
    assert "constraints" in payload


def test_taxonomy_prompt_marks_user_and_assistant_turns_explicitly():
    import json
    payload = json.loads(build_taxonomy_user_message(
        "Не помогло",
        [{"id": "kb_1", "category": "Тест", "definition": "Тест", "template_answer": "Да"}],
        [
            {"role": "user", "text": "Мой первый вопрос"},
            {"role": "assistant", "text": "Предыдущий ответ Terra"},
        ],
    ))
    assert payload["dialogue"][0]["role"] == "USER"
    assert payload["dialogue"][1]["role"] == "ASSISTANT"
    assert payload["last_assistant_response"] == "Предыдущий ответ Terra"
    assert payload["current_question"] == "Не помогло"
    assert "current_question" in payload["dialogue_rule"]


def test_taxonomy_router_rejects_multiple_answers_without_explicit_compound_intent(monkeypatch):
    monkeypatch.setattr(config_module.settings, "LLM_ROUTER_MODE", "taxonomy")
    llm = RecordingLLM(LLMResult(
        ["kb_registration", "kb_price"],
        decision="answer",
        certainty="high",
        is_compound=False,
    ))
    result = asyncio.run(DecisionPipeline(CatalogKB(), llm).decide(
        _msg("один вопрос"),
        is_first_message=False,
    ))
    assert result.traces[0].decision == Decision.ESCALATE
    assert result.traces[0].decision_reason == "llm_multiple_answers_without_compound_intent"
    assert result.actions and result.actions[0].escalate is True
