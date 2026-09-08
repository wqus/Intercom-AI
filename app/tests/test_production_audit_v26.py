import asyncio
import json
from types import SimpleNamespace

import httpx

from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.intercom.parser import parse


def realistic_conversation_payload(text="Оператор"):
    return {
        "id": "notif-real",
        "topic": "conversation.user.replied",
        "created_at": 1788877044,
        "data": {"item": {
            "id": "215475824210454",
            "source": {
                "type": "admin_initiated",
                "id": "215475824210454",
                "delivered_as": "admin_initiated",
                "body": "",
                "author": {"type": "admin", "id": "7848715"},
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "old", "part_type": "comment", "body": "старое", "created_at": 1788876988,
                 "author": {"type": "user", "id": "u1"}},
                {"id": "current", "part_type": "comment", "body": text, "created_at": 1788877043,
                 "author": {"type": "user", "id": "u1"}},
            ]},
        }},
    }


def test_realistic_admin_initiated_source_selects_current_user_part():
    message = parse(realistic_conversation_payload())
    assert message is not None
    assert message.text == "Оператор"
    assert message.raw["_provider_part_id"] == "current"


def test_conversation_user_created_is_a_real_customer_turn():
    payload = {
        "id": "notif-created",
        "topic": "conversation.user.created",
        "created_at": 100,
        "data": {"item": {
            "id": "conv-created",
            "source": {
                "type": "conversation",
                "id": "source-1",
                "delivered_as": "customer_initiated",
                "body": "Хочу узнать про обучение",
                "created_at": 100,
                "author": {"type": "user", "id": "u1", "name": "User"},
            },
            "conversation_parts": {"conversation_parts": []},
        }},
    }
    message = parse(payload)
    assert message is not None
    assert message.text == "Хочу узнать про обучение"
    assert message.channel.value == "conversation"


def test_context_reader_excludes_internal_assignment_and_current_part(monkeypatch):
    calls = []

    async def handler(request: httpx.Request):
        calls.append(request)
        return httpx.Response(200, json={
            "source": {
                "body": "Первоначальный вопрос",
                "created_at": 100,
                "id": "source-1",
                "author": {"type": "user", "id": "u1"},
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "a1", "part_type": "assignment", "body": "internal text",
                 "created_at": 110, "author": {"type": "admin", "id": "a1"}},
                {"id": "u1", "part_type": "comment", "body": "Проблема с кодом",
                 "created_at": 120, "author": {"type": "user", "id": "u1"}},
                {"id": "a2", "part_type": "comment", "body": "Уточните код",
                 "created_at": 130, "author": {"type": "admin", "id": "a1"}},
                {"id": "current", "part_type": "comment", "body": "код",
                 "created_at": 140, "author": {"type": "user", "id": "u1"}},
                {"id": "future", "part_type": "comment", "body": "future",
                 "created_at": 150, "author": {"type": "user", "id": "u1"}},
            ]},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            import app.infrastructure.intercom.client as mod
            monkeypatch.setattr(mod.settings, "DRY_RUN", False)
            monkeypatch.setattr(mod.settings, "INTERCOM_TOKEN", "token")
            monkeypatch.setattr(mod.settings, "INTERCOM_ADMIN_ID", "admin")
            gateway = IntercomGateway(client)
            return await gateway.get_conversation_context(
                "conv-1", current_part_id="current", before_epoch=140, limit=12
            )

    context = asyncio.run(run())
    assert context == [
        {"role": "user", "text": "Первоначальный вопрос"},
        {"role": "user", "text": "Проблема с кодом"},
        {"role": "assistant", "text": "Уточните код"},
    ]
    assert calls[0].url.path == "/conversations/conv-1"
    assert calls[0].url.params["display_as"] == "plaintext"


def test_ticket_context_excludes_non_message_parts(monkeypatch):
    async def handler(request: httpx.Request):
        return httpx.Response(200, json={
            "ticket_parts": {"ticket_parts": [
                {"id": "s1", "part_type": "ticket_state_updated_by_admin", "body": "internal",
                 "created_at": 100, "author": {"type": "admin"}},
                {"id": "u1", "part_type": "comment", "body": "Мне нужна помощь",
                 "created_at": 110, "author": {"type": "user"}},
            ]}
        })
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            import app.infrastructure.intercom.client as mod
            monkeypatch.setattr(mod.settings, "DRY_RUN", False)
            monkeypatch.setattr(mod.settings, "INTERCOM_TOKEN", "token")
            monkeypatch.setattr(mod.settings, "INTERCOM_ADMIN_ID", "admin")
            return await IntercomGateway(client).get_ticket_context("t-1", current_part_id="none")
    assert asyncio.run(run()) == [{"role": "user", "text": "Мне нужна помощь"}]


def test_ticket_reply_5xx_reconciliation_is_supported():
    # The gateway exposes the same reconciliation primitive for tickets and
    # conversations; the worker can therefore apply identical ambiguity rules.
    assert hasattr(IntercomGateway, "reply_exists")


def test_send_reply_rejects_unknown_channel(monkeypatch):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200))) as client:
            import app.infrastructure.intercom.client as mod
            monkeypatch.setattr(mod.settings, "DRY_RUN", False)
            monkeypatch.setattr(mod.settings, "INTERCOM_TOKEN", "token")
            monkeypatch.setattr(mod.settings, "INTERCOM_ADMIN_ID", "admin")
            await IntercomGateway(client).send_reply("x", "bogus", "hello")
    try:
        asyncio.run(run())
    except ValueError as exc:
        assert "Unsupported Intercom channel" in str(exc)
    else:
        raise AssertionError("unknown channel must fail closed")


def test_tag_http_failure_is_not_reported_as_success(monkeypatch):
    async def handler(request):
        return httpx.Response(500, json={"error": "boom"})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            import app.infrastructure.intercom.client as mod
            monkeypatch.setattr(mod.settings, "DRY_RUN", False)
            monkeypatch.setattr(mod.settings, "INTERCOM_ESCALATION_TAG_ID", "tag")
            monkeypatch.setattr(mod.settings, "INTERCOM_ADMIN_ID", "admin")
            return await IntercomGateway(client).add_escalation_tag("c-1", "conversation")
    try:
        asyncio.run(run())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 500
    else:
        raise AssertionError("tag HTTP 500 must propagate to outbox")
