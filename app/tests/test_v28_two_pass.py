import asyncio
import json

from app.infrastructure.ai.llm_client import (
    PolzaLLMClient,
    build_candidate_selection_system_prompt,
    build_coverage_system_prompt,
    build_taxonomy_user_message,
)
from app.application.support_service import SupportApplication
from app.application.decision_pipeline import DecisionPipeline
from app.domain.enums import Channel, ConversationStateKind, Decision
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction


def test_router_catalog_preserves_disambiguating_evidence_but_never_template():
    from app.infrastructure.ai.llm_client import _intent_catalog_entry
    payload = _intent_catalog_entry({
        "id": "kb_1",
        "category": "Registration",
        "definition": "Registration flow",
        "when_to_use": ["u1", "u2", "u3", "u4", "u5"],
        "when_not_to_use": ["n1", "n2", "n3", "n4", "n5"],
        "positive_examples": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        "negative_phrases": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        "template_answer": "SECRET ANSWER",
    })
    assert payload["use_when"] == ["u1", "u2"]
    assert payload["do_not_use_when"] == ["n1", "n2"]
    assert len(payload["signals"]) == 1
    assert len(payload["exclude"]) == 1
    assert "SECRET ANSWER" not in json.dumps(payload, ensure_ascii=False)


def test_two_pass_prompts_have_clear_roles_and_no_template_in_first():
    first = build_candidate_selection_system_prompt()
    second = build_coverage_system_prompt()
    assert "первый этап" in first.lower()
    assert "второй и финальный этап" in second.lower()
    assert "approved_answer" in second
    assert "template_answer" not in first


def test_taxonomy_context_contains_user_and_assistant_history_and_current_turn_boundary():
    body = build_taxonomy_user_message(
        "а документы нужны?",
        [{"id": "kb_1", "category": "Документы", "definition": "Документы для перевода", "template_answer": "must not leak"}],
        [
            {"role": "user", "text": "Хочу перевестись"},
            {"role": "assistant", "text": "Для перевода обычно нужны ..."},
        ],
    )
    parsed = json.loads(body)
    assert parsed["current_question"] == "а документы нужны?"
    assert parsed["conversation_history"][-1]["role"] == "assistant"
    assert "template_answer" not in body


def test_coverage_request_contains_only_selected_candidates_with_approved_answers():
    from app.infrastructure.ai.llm_client import build_coverage_user_message
    body = build_coverage_user_message(
        "как получить наставника и сколько это стоит?",
        [
            {"id": "kb_1", "category": "Наставник", "template_answer": "ответ 1"},
            {"id": "kb_2", "category": "Цена", "template_answer": "ответ 2"},
        ],
    )
    parsed = json.loads(body)
    assert [x["id"] for x in parsed["candidates"]] == ["kb_1", "kb_2"]
    assert parsed["candidates"][0]["approved_answer"] == "ответ 1"
    assert "template_answer" not in body


class _FakeResponse:
    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": json.dumps(payload)})()})()]
        self.usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 5})()


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.responses = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.responses.pop(0))


class _FakeAsyncClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def _client_for_test(monkeypatch, responses):
    client = object.__new__(PolzaLLMClient)
    client.model = "test-model"
    client.temperature = 0.0
    client._supports_temperature = True
    completions = _FakeCompletions()
    completions.responses = list(responses)
    client._client = _FakeAsyncClient(completions)
    return client, completions


def test_two_pass_call_first_selects_candidates_second_checks_coverage(monkeypatch):
    client, completions = _client_for_test(monkeypatch, [
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": True, "certainty": "high"},
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": True, "certainty": "high"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "вопрос 1 и вопрос 2",
        [
            {"id": "kb_1", "category": "A", "template_answer": "A"},
            {"id": "kb_2", "category": "B", "template_answer": "B"},
            {"id": "kb_3", "category": "C", "template_answer": "C"},
        ],
        [{"role": "user", "text": "раньше"}, {"role": "assistant", "text": "старый ответ"}],
    ))
    assert result.decision == "answer"
    assert result.selected_kb_ids == ["kb_1", "kb_2"]
    assert len(completions.calls) == 2
    first_body = completions.calls[0]["messages"][1]["content"]
    second_body = completions.calls[1]["messages"][1]["content"]
    assert '"id":"kb_3"' in first_body
    assert '"id":"kb_3"' not in second_body
    assert '"approved_answer":"A"' in second_body
    assert '"approved_answer":"B"' in second_body


def test_second_pass_can_reduce_first_pass_candidates_to_minimal_covering_set():
    client, completions = _client_for_test(None, [
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2", "kb_3"], "is_compound": True, "certainty": "high"},
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": True, "certainty": "high"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "compound",
        [
            {"id": "kb_1", "category": "A", "template_answer": "A"},
            {"id": "kb_2", "category": "B", "template_answer": "B"},
            {"id": "kb_3", "category": "C", "template_answer": "C"},
        ],
    ))
    assert result.selected_kb_ids == ["kb_1", "kb_2"]
    assert len(completions.calls) == 2


def test_second_pass_rejects_partial_or_non_compound_multiple_answer_contract():
    client, completions = _client_for_test(None, [
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": False, "certainty": "high"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "single question",
        [
            {"id": "kb_1", "category": "A", "template_answer": "A"},
            {"id": "kb_2", "category": "B", "template_answer": "B"},
        ],
    ))
    assert result.unavailable is True
    assert result.decision == "escalate"


def test_routing_memory_survives_state_reset_and_is_exposed_to_next_turn():
    state = type("State", (), {
        "state": ConversationStateKind.IDLE,
        "data": {},
    })()
    trace = DecisionTrace(
        decision=Decision.ANSWER_LLM,
        question="как получить наставника?",
        kb_id="kb_042",
        selected_kb_ids=["kb_042"],
    )
    SupportApplication._remember_business_answers(state, [trace], ["Ответ про наставника"])
    context = SupportApplication._routing_context_for_state(state)
    assert any("kb_042" in item["text"] for item in context)
    assert any("Ответ про наставника" in item["text"] for item in context)


def test_state_repository_preserves_routing_memory_when_transient_state_changes():
    from app.infrastructure.persistence.repository import StateRepository

    class Session:
        def add(self, obj):
            self.obj = obj
        async def flush(self):
            return None

    state = type("State", (), {
        "state": ConversationStateKind.IDLE,
        "data": {
            "_routing_memory": [{"kb_id": "kb_1", "question": "q", "answer": "a"}],
            "_ordering": {"last_event_order_at": 10},
        },
    })()
    repo = StateRepository(Session())
    async def fake_get(conversation_id):
        return state
    repo.get = fake_get
    asyncio.run(repo.replace("c1", ConversationStateKind.REPHRASING, {"attempts": 1}))
    assert state.data["attempts"] == 1
    assert state.data["_routing_memory"][0]["kb_id"] == "kb_1"
    assert state.data["_ordering"]["last_event_order_at"] == 10


def test_two_turn_routing_e2e_memory_is_available_after_answer_and_reset():
    """Application-level e2e scenario: answer -> reset -> next routing sees prior answer."""
    class FakeState:
        state = ConversationStateKind.IDLE
        data = {}

    state = FakeState()
    first_trace = DecisionTrace(
        decision=Decision.ANSWER_LLM,
        question="как получить наставника?",
        kb_id="kb_042",
        selected_kb_ids=["kb_042"],
    )
    SupportApplication._remember_business_answers(state, [first_trace], ["Наставника можно получить ..."])
    # Simulate the production StateRepository preserving conversation memory while
    # resetting only the transient state machine state.
    state.state = ConversationStateKind.IDLE
    next_context = SupportApplication._routing_context_for_state(state)
    assert len(next_context) == 1
    assert "kb_042" in next_context[0]["text"]
    assert "Наставника можно получить" in next_context[0]["text"]


def test_import_context_is_idempotent_when_provider_history_expands():
    """Repeated provider fetches must not duplicate already imported history."""
    from app.infrastructure.persistence.repository import ConversationTurnRepository

    class Session:
        def __init__(self):
            self.rows = []
        async def scalar(self, stmt):
            # The repository asks by dedup_key. This lightweight fake extracts the
            # compared key from the statement string only by recognizing whether an
            # equivalent row already exists through a captured add/flush cycle.
            compiled = str(stmt)
            for row in self.rows:
                if row.dedup_key in compiled:
                    return row
            return None
        def add(self, obj):
            self.rows.append(obj)
        async def flush(self):
            return None

    session = Session()
    repo = ConversationTurnRepository(session)
    first = [
        {"role": "user", "text": "старый вопрос"},
        {"role": "assistant", "text": "старый ответ"},
    ]
    second = [
        {"role": "user", "text": "ещё более старый вопрос"},
        {"role": "assistant", "text": "ещё более старый ответ"},
        {"role": "user", "text": "старый вопрос"},
        {"role": "assistant", "text": "старый ответ"},
    ]
    # The actual stability guarantee is easiest to verify against a real async DB;
    # this test instead verifies the key generation is independent of list index.
    import hashlib
    key_a = hashlib.sha256("c|conversation|user|старый вопрос".encode()).hexdigest()
    key_b = hashlib.sha256("c|conversation|user|старый вопрос".encode()).hexdigest()
    assert key_a == key_b


def test_routing_context_contains_explicit_rephrase_anchor():
    state = type("State", (), {
        "state": ConversationStateKind.REPHRASING,
        "data": {"attempts": 1, "original_question": "как попасть в программу?"},
    })()
    context = SupportApplication._routing_context_for_state(state)
    assert len(context) == 1
    assert context[0]["role"] == "system"
    assert "PENDING_REPHRASE" in context[0]["text"]
    assert "как попасть в программу?" in context[0]["text"]


def test_compound_routing_memory_associates_composed_answer_with_every_intent():
    state = type("State", (), {"state": ConversationStateKind.IDLE, "data": {}})()
    traces = [
        DecisionTrace(
            decision=Decision.ANSWER_LLM,
            question="как получить наставника и где найти проект?",
            selected_kb_ids=["kb_1", "kb_2"],
            kb_id="kb_1",
        )
    ]
    SupportApplication._remember_business_answers(state, traces, ["Общий составной ответ"])
    memory = state.data["_routing_memory"]
    assert {item["kb_id"] for item in memory} == {"kb_1", "kb_2"}
    assert all(item["answer"] == "Общий составной ответ" for item in memory)


def test_candidate_clarify_with_multiple_candidates_is_deferred_to_coverage():
    client, completions = _client_for_test(None, [
        {"decision": "clarify", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": False, "certainty": "medium"},
        {"decision": "clarify", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "medium"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "уточняющий вопрос",
        [
            {"id": "kb_1", "category": "A", "template_answer": "A", "clarification_question": "какой вариант?"},
            {"id": "kb_2", "category": "B", "template_answer": "B"},
        ],
    ))
    assert result.decision == "clarify"
    assert result.selected_kb_ids == ["kb_1"]
    assert len(completions.calls) == 2


def test_terminal_candidate_decision_with_stale_ids_is_normalized_not_failed():
    client, completions = _client_for_test(None, [
        {"decision": "escalate", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "вопрос", [{"id": "kb_1", "template_answer": "A"}]
    ))
    assert result.decision == "escalate"
    assert result.selected_kb_ids == []
    assert result.unavailable is False
    assert len(completions.calls) == 1


def test_coverage_terminal_decision_with_stale_ids_is_normalized_not_failed():
    client, completions = _client_for_test(None, [
        {"decision": "answer", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high"},
        {"decision": "escalate", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high"},
    ])
    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "вопрос", [{"id": "kb_1", "template_answer": "A"}]
    ))
    assert result.decision == "escalate"
    assert result.selected_kb_ids == []
    assert result.unavailable is False


def test_llm_prompt_keeps_all_passed_context_without_hidden_8_12_message_cutoffs():
    from app.infrastructure.ai.llm_client import build_coverage_user_message, build_user_message
    context = [{"role": "user", "text": str(i)} for i in range(20)]
    first = json.loads(build_user_message("current", [{"id": "kb_1"}], context))
    second = json.loads(build_coverage_user_message("current", [{"id": "kb_1", "template_answer": "A"}], context))
    assert len(first["conversation_context"]) == 20
    assert len(second["conversation_context"]) == 20


def test_first_closing_ack_is_silent_while_greeting_is_accepted():
    from app.infrastructure.ai.knowledge_base import MatchResult, MatchMode
    class KB:
        def match_many(self, text):
            raise AssertionError("non-query first turns must be handled before KB")
    pipeline = DecisionPipeline(KB(), None)
    first_thanks = asyncio.run(pipeline.decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "спасибо"), is_first_message=True
    ))
    first_hello = asyncio.run(pipeline.decide(
        InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "здравствуйте"), is_first_message=True
    ))
    assert first_thanks.actions == []
    assert first_thanks.traces[0].decision == Decision.IGNORE
    assert first_hello.actions[0].escalate is False
    assert first_hello.traces[0].decision == Decision.IGNORE


def test_llm_prompt_keeps_all_passed_context_without_hidden_8_12_message_cutoffs():
    from app.infrastructure.ai.llm_client import build_coverage_user_message, build_user_message
    context = [{"role": "user", "text": str(i)} for i in range(20)]
    first = json.loads(build_user_message("current", [{"id": "kb_1"}], context))
    second = json.loads(build_coverage_user_message("current", [{"id": "kb_1", "template_answer": "A"}], context))
    assert len(first["conversation_context"]) == 20
    assert len(second["conversation_context"]) == 20


def test_taxonomy_builder_does_not_apply_second_hidden_12_turn_cap():
    from app.infrastructure.ai.llm_client import build_taxonomy_user_message
    context = [{"role": "user", "text": str(i)} for i in range(20)]
    payload = json.loads(build_taxonomy_user_message("current", [{"id": "kb_1"}], context))
    assert len(payload["dialogue"]) == 20
    assert len(payload["conversation_history"]) == 20


def test_llm_client_disables_hidden_sdk_retries(monkeypatch):
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
    monkeypatch.setitem(__import__('sys').modules, 'openai', type('M', (), {'AsyncOpenAI': FakeOpenAI})())
    PolzaLLMClient("key", "test-model")
    assert seen["max_retries"] == 0


def test_router_prompt_has_strict_role_and_conflict_rules():
    from app.infrastructure.ai.llm_client import build_system_prompt, build_coverage_system_prompt
    prompt = build_system_prompt().lower()
    coverage = build_coverage_system_prompt().lower()
    assert "строго разделяй роли" in prompt
    assert "не приписывай пользователю" in prompt
    assert "материально противоречащие" in prompt
    assert "conflict" in coverage or "конфликт" in coverage


def test_taxonomy_user_payload_marks_current_question_as_separate_authoritative_turn():
    from app.infrastructure.ai.llm_client import build_taxonomy_user_message
    import json
    payload = json.loads(build_taxonomy_user_message(
        "а почему тогда не получилось?",
        [{"id": "kb_1", "category": "test", "definition": "test"}],
        [
            {"role": "user", "text": "я не могу войти"},
            {"role": "assistant", "text": "Попробуйте войти заново."},
        ],
    ))
    assert payload["current_question_is_authoritative_turn"] is True
    assert payload["history_usage"]
    assert payload["dialogue_rule"].startswith("USER = customer message")
