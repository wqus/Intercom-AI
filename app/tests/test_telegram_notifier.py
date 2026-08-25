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
