import asyncio
import json
from types import SimpleNamespace
from app.infrastructure.ai.llm_client import PolzaLLMClient, LLMResult


def test_llm_result_contract():
    result = LLMResult(["kb_1", "kb_2"])
    assert result.selected_kb_ids == ["kb_1", "kb_2"]
    assert result.escalation is False


def test_polza_client_schema_is_dynamic(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"result": ["kb_2"]})))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("не могу записаться", [{"id":"kb_1"},{"id":"kb_2"}]))
    assert result.selected_kb_ids == ["kb_2"]
    assert result.certainty == "high"
    schema = captured["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["decision"]["enum"] == ["answer", "clarify", "ignore", "escalate"]
    assert schema["properties"]["certainty"]["enum"] == ["high", "medium", "low"]
    assert schema["properties"]["selected_kb_ids"]["items"]["enum"] == ["kb_1", "kb_2"]
    assert "answer" not in schema["properties"]
    assert "reason" not in schema["properties"]


def test_polza_client_rejects_mixed_escalation(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"result": ["escalation", "kb_1"]})))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id":"kb_1"}]))
    assert result.unavailable is True
    assert result.decision == "escalate"
    assert result.selected_kb_ids == []


def test_polza_client_sends_top10_and_compact_payload_for_sonnet5(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"result": ["kb_3"]})))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    candidates = [{"id": f"kb_{i}", "category": f"Категория {i}", "definition": f"Определение {i}", "when_not_to_use": ["Не подходит для соседнего intent"] , "template_answer": "Скрытый шаблон", "priority": 99} for i in range(1, 10)]
    result = asyncio.run(client.classify("q", candidates))
    assert result.selected_kb_ids == ["kb_3"]
    payload = json.loads(captured["messages"][1]["content"])
    assert [x["id"] for x in payload["candidates"]] == [f"kb_{i}" for i in range(1, 10)]
    assert "temperature" not in captured
    assert captured["max_tokens"] == 768


def test_polza_client_marks_transport_or_empty_output_as_unavailable(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id": "kb_1", "category": "x"}]))
    assert result.unavailable is True
    assert result.decision == "escalate"
    assert result.selected_kb_ids == []


def test_polza_client_parses_markdown_wrapped_json(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="```json\n{\"result\":[\"kb_1\"]}\n```")
                )]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id": "kb_1"}]))
    assert result.selected_kb_ids == ["kb_1"]
    assert result.unavailable is False


def test_polza_client_retries_when_first_content_is_empty(monkeypatch):
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            content = "" if len(calls) == 1 else json.dumps({"result": ["kb_1"]})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id": "kb_1"}]))
    assert result.selected_kb_ids == ["kb_1"]
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 768
    assert calls[1]["max_tokens"] == 1536


def test_polza_client_extracts_text_parts(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content=[{"type": "text", "text": "{\"result\":[\"kb_1\"]}"}]
                    )
                )]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id": "kb_1"}]))
    assert result.selected_kb_ids == ["kb_1"]


def test_polza_client_retries_when_first_content_is_malformed(monkeypatch):
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            content = "not-json" if len(calls) == 1 else json.dumps({"result": ["kb_1"]})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "anthropic/claude-sonnet-5")
    result = asyncio.run(client.classify("q", [{"id": "kb_1"}]))
    assert result.selected_kb_ids == ["kb_1"]
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == 1536


def test_luna_pro_uses_structured_router_schema_and_omits_temperature(monkeypatch):
    captured = {}
    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            payload = {"decision": "answer", "selected_kb_ids": ["kb_1"], "is_compound": False, "certainty": "high"}
            message = SimpleNamespace(content=json.dumps(payload))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys, types
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "openai/gpt-5.6-luna-pro")
    result = asyncio.run(client.classify(
        "код",
        [{"id": "kb_1", "category": "Регистрация", "when_to_use": ["Проблемы с регистрацией"]}],
        [{"role": "user", "text": "Не могу зарегистрироваться"}],
        taxonomy_mode=True,
    ))
    assert result.decision == "answer"
    assert result.certainty == "high"
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["certainty"]
    assert "temperature" not in captured
    # Taxonomy mode is intentionally two-pass: the first call receives the full
    # taxonomy/history, the second receives only the selected candidates and answers.
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["current_question"] == "код"
    assert payload["candidates"][0]["id"] == "kb_1"
    assert payload["candidates"][0]["approved_answer"] == ""
    assert captured["response_format"]["json_schema"]["name"] == "terra_answer_coverage"



def test_taxonomy_two_pass_passes_approved_answers_and_caps_coverage_at_four(monkeypatch):
    import sys, types
    calls = []
    responses = [
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2", "kb_3", "kb_4"], "is_compound": True, "certainty": "high"},
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_3"], "is_compound": True, "certainty": "high"},
    ]

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            payload = responses[len(calls) - 1]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "openai/gpt-5.6-luna-pro")
    catalog = [
        {"id": f"kb_{i}", "category": f"Категория {i}", "definition": "D", "template_answer": f"Ответ {i}", "when_to_use": ["X"]}
        for i in range(1, 7)
    ]
    result = asyncio.run(client.classify("compound", catalog, taxonomy_mode=True))

    assert result.decision == "answer"
    assert result.selected_kb_ids == ["kb_1", "kb_3"]
    assert len(calls) == 2

    first_payload = json.loads(calls[0]["messages"][1]["content"])
    assert len(first_payload["intent_catalog"]) == 6
    first_schema = calls[0]["response_format"]["json_schema"]["schema"]
    assert first_schema["properties"]["selected_kb_ids"]["maxItems"] == 4

    coverage_payload = json.loads(calls[1]["messages"][1]["content"])
    assert len(coverage_payload["candidates"]) == 4
    assert {item["id"] for item in coverage_payload["candidates"]} == {"kb_1", "kb_2", "kb_3", "kb_4"}
    assert coverage_payload["candidates"][0]["approved_answer"] == "Ответ 1"
    coverage_schema = calls[1]["response_format"]["json_schema"]["schema"]
    assert coverage_schema["properties"]["selected_kb_ids"]["maxItems"] == 4


def test_taxonomy_two_pass_fails_closed_when_coverage_is_partial(monkeypatch):
    import sys, types
    responses = [
        {"decision": "answer", "selected_kb_ids": ["kb_1", "kb_2"], "is_compound": True, "certainty": "high"},
        {"decision": "escalate", "selected_kb_ids": [], "is_compound": False, "certainty": "high"},
    ]
    index = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal index
            payload = responses[index]
            index += 1
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    client = PolzaLLMClient("key", "openai/gpt-5.6-luna-pro")
    result = asyncio.run(client.classify(
        "compound",
        [
            {"id": "kb_1", "category": "A", "template_answer": "Ответ A"},
            {"id": "kb_2", "category": "B", "template_answer": "Ответ B"},
        ],
        taxonomy_mode=True,
    ))
    assert result.decision == "escalate"
    assert result.selected_kb_ids == []
