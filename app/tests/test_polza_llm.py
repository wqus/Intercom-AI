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
    assert schema["properties"]["decision"]["enum"] == ["answer", "clarify", "escalate"]
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
            payload = {"decision": "answer", "selected_kb_ids": ["kb_1"], "certainty": "high"}
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
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["conversation_history"][0]["text"] == "Не могу зарегистрироваться"

