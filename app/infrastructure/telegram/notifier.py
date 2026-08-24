from __future__ import annotations

import logging
import httpx

from app.core.config import settings

logger = logging.getLogger("telegram")


class TelegramNotifier:
    """Best-effort operator notification. Notification failures must never retry a support event."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def notify_escalation(
        self,
        *,
        entity_id: str,
        channel: str,
        reason: str,
        text: str,
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

        body = text.strip().replace("\x00", "")
        if len(body) > 1500:
            body = body[:1497] + "..."
        message = (
            "🚨 Эскалация на оператора\n"
            f"Канал: {channel}\n"
            f"ID: {entity_id}\n"
            f"Причина: {reason}\n\n"
            f"Сообщение пользователя:\n{body}"
        )
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            response = await self.client.post(
                url,
                json={"chat_id": chat_id, "text": message},
            )
            response.raise_for_status()
            logger.info("Operator notification sent entity=%s reason=%s", entity_id, reason)
            return True
        except httpx.HTTPError as exc:
            logger.warning("Operator notification failed entity=%s: %s", entity_id, exc)
            return False
