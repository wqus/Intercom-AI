import asyncio
import json

from app.infrastructure.ai.llm_client import (
    PolzaLLMClient,
    build_candidate_selection_user_message,
    build_coverage_user_message,
)


def test_first_and_second_pass_receive_identical_canonical_context():
    context = [
        {"role": "user", "text": "старый вопрос"},
        {"role": "assistant", "text": "старый ответ"},
        {"role": "system", "text": "PENDING_CLARIFICATION: kb_id=kb_1; attempts=1"},
    ]
    first = json.loads(build_candidate_selection_user_message(
        "а теперь почему?", [{"id": "kb_1", "template_answer": "Ответ"}], context
    ))
    second = json.loads(build_coverage_user_message(
        "а теперь почему?", [{"id": "kb_1", "template_answer": "Ответ"}], context
    ))
    for key in ("conversation_history", "pending_state", "last_assistant_response", "current_question", "context_contract"):
        assert first[key] == second[key]
    assert first["current_question"] == "а теперь почему?"
    assert first["conversation_history"][-1]["role"] == "assistant"
    assert first["pending_state"]


def test_first_pass_escalate_with_candidates_is_not_allowed_to_skip_coverage():
    class FakeCompletions:
        def __init__(self):
            self.calls = []
            self.payloads = [
                {"decision": "escalate", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "medium"},
                {"decision": "answer", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high", "coverage_status": "complete", "coverage_gaps": []},
            ]
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = self.payloads.pop(0)
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()

    completions = FakeCompletions()
    client = object.__new__(PolzaLLMClient)
    client.model = "test-model"
    client._supports_temperature = True
    client._client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "вопрос", [{"id": "kb_1", "template_answer": "Ответ"}]
    ))
    assert result.decision == "answer"
    assert result.selected_kb_ids == ["kb_1"]
    assert len(completions.calls) == 2


def test_second_pass_partial_claim_is_forced_to_escalation():
    class FakeCompletions:
        def __init__(self):
            self.calls = 0
        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                payload = {"decision": "answer", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high"}
            else:
                payload = {"decision": "answer", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high", "coverage_status": "partial", "coverage_gaps": ["Не покрыт вопрос пользователя"]}
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()

    completions = FakeCompletions()
    client = object.__new__(PolzaLLMClient)
    client.model = "test-model"
    client._supports_temperature = True
    client._client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

    result = asyncio.run(client._classify_taxonomy_with_coverage(
        "вопрос", [{"id": "kb_1", "template_answer": "Ответ"}]
    ))
    assert result.decision == "escalate"
    assert result.escalation is True
    assert result.selected_kb_ids == []


def test_full_answer_material_is_available_to_first_router_catalog():
    body = build_candidate_selection_user_message(
        "вопрос", [{"id": "kb_1", "category": "A", "template_answer": "Approved answer"}], []
    )
    assert '"approved_answer":"Approved answer"' in body
