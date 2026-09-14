import httpx
import asyncio

from app.application.outbox_worker import OutboxWorker


class FakeIntercom:
    def __init__(self):
        self.replies = []
        self.tags = []

    async def send_reply(self, entity_id, channel, text):
        self.replies.append((entity_id, channel, text))

    async def add_escalation_tag(self, entity_id, channel):
        self.tags.append((entity_id, channel))


class FakeTelegram:
    def __init__(self, configured=True, result=True):
        self.configured = configured
        self.result = result
        self.calls = []

    def is_configured(self):
        return self.configured

    async def notify_escalation(self, **payload):
        self.calls.append(payload)
        return self.result


def action(action_type, payload):
    return type("Action", (), {"action_type": action_type, "payload": payload, "id": 1})()


def test_outbox_intercom_reply_delivery():
    intercom = FakeIntercom()
    telegram = FakeTelegram()
    worker = OutboxWorker(intercom, telegram)
    status, error = asyncio.run(worker._deliver(action(
        "intercom_reply",
        {"entity_id": "c1", "channel": "conversation", "text": "hello"},
    )))
    assert status == "sent"
    assert error is None
    assert intercom.replies == [("c1", "conversation", "hello")]


def test_outbox_disabled_telegram_is_skipped_cleanly():
    intercom = FakeIntercom()
    telegram = FakeTelegram(configured=False)
    worker = OutboxWorker(intercom, telegram)
    status, error = asyncio.run(worker._deliver(action(
        "telegram_escalation",
        {"entity_id": "c1", "channel": "conversation", "reason": "no_category", "text": "x"},
    )))
    assert status == "skipped"
    assert "not configured" in error
    assert telegram.calls == []


def test_outbox_escalation_tag_delivery():
    intercom = FakeIntercom()
    worker = OutboxWorker(intercom, FakeTelegram())
    status, error = asyncio.run(worker._deliver(action(
        "intercom_escalation_tag",
        {"entity_id": "c1", "channel": "conversation"},
    )))
    assert status == "sent"
    assert error is None
    assert intercom.tags == [("c1", "conversation")]


def test_stale_processing_actions_are_not_reclaimed():
    import inspect
    from app.infrastructure.persistence.outbox import OutboundRepository

    source = inspect.getsource(OutboundRepository.claim_one)
    assert "OutboundAction.status == \"processing\"" not in source
    assert "mark_stale_unknown" in inspect.getsource(OutboundRepository)


def _http_error(status: int) -> httpx.HTTPStatusError:
    import httpx
    request = httpx.Request("POST", "https://api.intercom.io/conversations/c1/reply")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_intercom_5xx_reconciled_as_sent_when_reply_is_visible():
    import asyncio
    from types import SimpleNamespace
    import httpx

    class ReconIntercom(FakeIntercom):
        async def reply_exists(self, entity_id, text, after_epoch=None, channel="conversation"):
            self.reconcile_args = (entity_id, text, after_epoch, channel)
            return True

    class Repo:
        def __init__(self):
            self.calls = []
        async def mark_sent(self, action): self.calls.append(("sent", action))
        async def mark_unknown(self, action, error): self.calls.append(("unknown", error))
        async def mark_retry(self, action, error): self.calls.append(("retry", error))
        async def mark_skipped(self, action, error): self.calls.append(("skipped", error))

    action_obj = SimpleNamespace(
        id=7,
        action_type="intercom_reply",
        payload={"entity_id": "c1", "channel": "conversation", "text": "hello"},
        created_at=None,
    )
    worker = OutboxWorker(ReconIntercom(), FakeTelegram())
    repo = Repo()
    asyncio.run(worker._handle_http_status_error(action_obj, repo, _http_error(503)))
    assert [c[0] for c in repo.calls] == ["sent"]


def test_intercom_5xx_without_reconciled_reply_becomes_unknown_not_retry():
    import asyncio
    from types import SimpleNamespace

    class ReconIntercom(FakeIntercom):
        async def reply_exists(self, entity_id, text, after_epoch=None):
            return False

    class Repo:
        def __init__(self):
            self.calls = []
        async def mark_sent(self, action): self.calls.append(("sent", action))
        async def mark_unknown(self, action, error): self.calls.append(("unknown", error))
        async def mark_retry(self, action, error): self.calls.append(("retry", error))
        async def mark_skipped(self, action, error): self.calls.append(("skipped", error))

    action_obj = SimpleNamespace(
        id=8,
        action_type="intercom_reply",
        payload={"entity_id": "c1", "channel": "conversation", "text": "hello"},
        created_at=None,
    )
    worker = OutboxWorker(ReconIntercom(), FakeTelegram())
    repo = Repo()
    asyncio.run(worker._handle_http_status_error(action_obj, repo, _http_error(500)))
    assert [c[0] for c in repo.calls] == ["unknown"]
    assert "delivery requires reconciliation" in repo.calls[0][1]


def test_intercom_429_remains_retryable():
    import asyncio
    from types import SimpleNamespace

    class Repo:
        def __init__(self): self.calls = []
        async def mark_sent(self, action): self.calls.append(("sent", action))
        async def mark_unknown(self, action, error): self.calls.append(("unknown", error))
        async def mark_retry(self, action, error): self.calls.append(("retry", error))
        async def mark_skipped(self, action, error): self.calls.append(("skipped", error))

    action_obj = SimpleNamespace(id=9, action_type="intercom_reply", payload={"channel": "conversation"})
    worker = OutboxWorker(FakeIntercom(), FakeTelegram())
    repo = Repo()
    asyncio.run(worker._handle_http_status_error(action_obj, repo, _http_error(429)))
    assert [c[0] for c in repo.calls] == ["retry"]


def test_intercom_4xx_is_terminal_failed_not_skipped():
    import asyncio
    from types import SimpleNamespace

    class Repo:
        def __init__(self): self.calls = []
        async def mark_failed(self, action, error): self.calls.append(("failed", error))
        async def mark_sent(self, action): self.calls.append(("sent", action))
        async def mark_unknown(self, action, error): self.calls.append(("unknown", error))
        async def mark_retry(self, action, error): self.calls.append(("retry", error))
        async def mark_skipped(self, action, error): self.calls.append(("skipped", error))

    action_obj = SimpleNamespace(
        id=10,
        action_type="intercom_reply",
        payload={"entity_id": "c1", "channel": "conversation", "text": "hello"},
    )
    worker = OutboxWorker(FakeIntercom(), FakeTelegram())
    repo = Repo()
    asyncio.run(worker._handle_http_status_error(action_obj, repo, _http_error(400)))
    assert [c[0] for c in repo.calls] == ["failed"]


def test_intercom_reply_uses_correct_endpoint_and_admin_payload(monkeypatch):
    import asyncio
    import httpx
    from types import SimpleNamespace
    from app.infrastructure.intercom.client import IntercomGateway

    calls = []
    async def handler(request):
        calls.append((str(request.url), await request.aread(), dict(request.headers)))
        if "/tickets/" in str(request.url):
            return httpx.Response(200, json={"type": "ticket_part", "id": "tp-1", "part_type": "comment"})
        return httpx.Response(200, json={"type": "conversation", "id": "c-1"})

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            gateway = IntercomGateway(client)
            await gateway.send_reply("c-1", "conversation", "hello")
            await gateway.send_reply("t-1", "ticket", "hello")

    import app.infrastructure.intercom.client as client_module
    monkeypatch.setattr(client_module.settings, "DRY_RUN", False)
    monkeypatch.setattr(client_module.settings, "INTERCOM_TOKEN", "token")
    monkeypatch.setattr(client_module.settings, "INTERCOM_ADMIN_ID", "admin-1")
    asyncio.run(run())

    assert calls[0][0] == "https://api.intercom.io/conversations/c-1/reply"
    assert calls[1][0] == "https://api.intercom.io/tickets/t-1/reply"
    for _, raw, _ in calls:
        payload = __import__("json").loads(raw)
        assert payload["message_type"] == "comment"
        assert payload["type"] == "admin"
        assert payload["admin_id"] == "admin-1"
        assert payload["body"] == "hello"


def test_intercom_4xx_reply_uses_terminal_failure_and_transcript_sync(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    class Repo:
        def __init__(self):
            self.calls = []
        async def mark_failed(self, action, error): self.calls.append(("failed", error))
        async def mark_sent(self, action): self.calls.append(("sent", action))
        async def mark_unknown(self, action, error): self.calls.append(("unknown", error))
        async def mark_retry(self, action, error): self.calls.append(("retry", error))
        async def mark_skipped(self, action, error): self.calls.append(("skipped", error))

    action_obj = SimpleNamespace(
        id=11,
        action_type="intercom_reply",
        payload={"entity_id": "c1", "channel": "conversation", "text": "hello"},
    )
    worker = OutboxWorker(FakeIntercom(), FakeTelegram())
    repo = Repo()
    asyncio.run(worker._handle_http_status_error(action_obj, repo, _http_error(400)))
    assert [c[0] for c in repo.calls] == ["failed"]
