from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger("telegram")


def _format_timestamp(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "—"

    # Intercom commonly provides Unix seconds for updated_at.
    if raw.isdigit():
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return raw

    # Preserve already-readable timestamps while normalizing ISO strings.
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def _escape(value: str) -> str:
    return html.escape(str(value or "—"), quote=False)


class TelegramNotifier:
    """Best-effort operator notification with Telegram-native HTML formatting."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    def is_configured(self) -> bool:
        return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_GROUP_ID)

    async def notify_escalation(
        self,
        *,
        entity_id: str,
        channel: str,
        reason: str,
        text: str,
        user_name: str = "",
        user_email: str = "",
        updated_at: str = "",
    ) -> bool:
        token = settings.TELEGRAM_BOT_TOKEN
        chat_id = settings.TELEGRAM_GROUP_ID
        if settings.DRY_RUN:
            logger.info(
                "[DRY RUN] operator notification entity=%s channel=%s reason=%s",
                entity_id,
                channel,
                reason,
            )
            return True
        if not token or not chat_id:
            logger.warning("Telegram notification skipped: TELEGRAM_BOT_TOKEN/TELEGRAM_GROUP_ID not configured")
            return False

        body = (text or "").strip().replace("\x00", "")
        # Keep the whole notification safely below Telegram's sendMessage limit.
        if len(body) > 2200:
            body = body[:2197] + "..."

        channel_label = "диалог" if channel == "conversation" else channel
        message = (
            "<b>Запрос пользователя был передан оператору</b>\n\n"
            f"<b>ID диалога:</b> {_escape(entity_id)}\n"
            f"<b>Тип:</b> {_escape(channel_label)}\n"
            f"<b>Причина эскалации:</b> {_escape(reason)}\n"
            f"<b>Имя:</b> {_escape(user_name)}\n"
            f"<b>Email:</b> {_escape(user_email)}\n"
            f"<b>Время последнего сообщения:</b> {_escape(_format_timestamp(updated_at))}\n\n"
            "<b>Текст обращения:</b>\n"
            f"<blockquote>{_escape(body)}</blockquote>"
        )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            response = await self.client.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            response.raise_for_status()
            logger.info("Operator notification sent entity=%s reason=%s", entity_id, reason)
            return True
        except httpx.HTTPError as exc:
            logger.warning("Operator notification failed entity=%s: %s", entity_id, exc)
            return False
