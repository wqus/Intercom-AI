import asyncio

from app.infrastructure.telegram.notifier import TelegramNotifier


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeClient:
    def __init__(self):
        self.kwargs = None

    async def post(self, url, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


def test_escalation_notification_uses_html_blockquote(monkeypatch):
    import app.infrastructure.telegram.notifier as module
    monkeypatch.setattr(module.settings, "DRY_RUN", False)
    monkeypatch.setattr(module.settings, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(module.settings, "TELEGRAM_GROUP_ID", "123")

    client = FakeClient()
    notifier = TelegramNotifier(client)
    asyncio.run(notifier.notify_escalation(
        entity_id="215475614754532",
        channel="conversation",
        reason="attachments",
        text="Спасибо <тест> & ещё",
        user_name="Ефремкина Анастасия",
        user_email="a.efremkina@mail.ru",
        updated_at="2026-08-24T18:31:54+00:00",
    ))

    body = client.kwargs["json"]
    assert body["parse_mode"] == "HTML"
    assert "<blockquote>Спасибо &lt;тест&gt; &amp; ещё</blockquote>" in body["text"]
    assert "<b>Запрос пользователя был передан оператору</b>" in body["text"]
    assert "<b>Имя:</b> Ефремкина Анастасия" in body["text"]


def test_telegram_http_error_does_not_log_bot_token(monkeypatch, caplog):
    import asyncio
    import httpx
    import logging
    import app.infrastructure.telegram.notifier as module

    token = "123456:SECRET_BOT_TOKEN"
    monkeypatch.setattr(module.settings, "DRY_RUN", False)
    monkeypatch.setattr(module.settings, "TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(module.settings, "TELEGRAM_GROUP_ID", "123")

    class FailingClient:
        async def post(self, url, **kwargs):
            request = httpx.Request("POST", url)
            raise httpx.ConnectError(f"connect failed for {url}", request=request)

    notifier = TelegramNotifier(FailingClient())
    with caplog.at_level(logging.WARNING, logger="telegram"):
        result = asyncio.run(notifier.notify_escalation(entity_id="c1", channel="conversation", reason="x", text="hello"))
    assert result is False
    assert token not in caplog.text
