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
