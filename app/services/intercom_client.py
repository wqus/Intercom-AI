import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("intercom_client")

INTERCOM_API_BASE = "https://api.intercom.io"


class IntercomClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers=self._headers(),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.INTERCOM_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Intercom-Version": "2.15",
        }

    async def close(self) -> None:
        await self._client.aclose()

    async def _add_conversation_part(
            self,
            conversation_id: str,
            message_type: str,
            body: Optional[str] = None,
            admin_id: Optional[str] = None,
            **extra: dict,
    ) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] %s conv=%s", message_type, conversation_id)
            return True

        admin_id = admin_id or settings.INTERCOM_ADMIN_ID
        if not admin_id:
            logger.error("INTERCOM_ADMIN_ID не настроен")
            return False
        if not settings.INTERCOM_TOKEN:
            logger.error("Missing INTERCOM_TOKEN")
            return False

        payload = {
            "message_type": message_type,
            "type": "admin",
            "admin_id": admin_id,
            **extra,
        }
        if body is not None:
            payload["body"] = body

        try:
            resp = await self._client.post(
                f"{INTERCOM_API_BASE}/conversations/{conversation_id}/parts",
                json=payload,
            )
            resp.raise_for_status()

            logger.info("%s успешно выполнен для conv=%s", message_type.upper(), conversation_id)
            return True

        except httpx.HTTPStatusError as e:
            logger.error(
                "%s error %s: %s",
                message_type.upper(),
                e.response.status_code,
                e.response.text[:300],
            )
            return False
        except httpx.RequestError as e:
            logger.error("Network error %s conv=%s: %s", message_type.upper(), conversation_id, e)
            return False

    async def send_conversations_reply(
            self,
            conversation_id: str,
            text: str,
            admin_id: Optional[str] = None,
    ) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] reply conv=%s", conversation_id)
            return True

        admin_id = admin_id or settings.INTERCOM_ADMIN_ID
        if not settings.INTERCOM_TOKEN:
            logger.error("Missing INTERCOM_TOKEN")
            return False

        try:
            resp = await self._client.post(
                f"{INTERCOM_API_BASE}/conversations/{conversation_id}/reply",
                json={
                    "message_type": "comment",
                    "type": "admin",
                    "admin_id": admin_id,
                    "body": text,
                },
            )
            resp.raise_for_status()
            logger.info("Reply sent to conv=%s", conversation_id)
            return True

        except httpx.HTTPStatusError as e:
            logger.error("Reply error %s: %s", e.response.status_code, e.response.text[:250])
            return False
        except httpx.RequestError as e:
            logger.error("Network error reply conv=%s: %s", conversation_id, e)
            return False

    async def send_ticket_reply(
            self,
            ticket_id: str,
            text: str,
            admin_id: Optional[str] = None,
    ) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] ticket reply id=%s", ticket_id)
            return True

        admin_id = admin_id or settings.INTERCOM_ADMIN_ID
        if not settings.INTERCOM_TOKEN:
            logger.error("Missing INTERCOM_TOKEN")
            return False

        try:
            resp = await self._client.post(
                f"{INTERCOM_API_BASE}/tickets/{ticket_id}/reply",
                json={
                    "message_type": "comment",
                    "type": "admin",
                    "admin_id": admin_id,
                    "body": text,
                },
            )
            resp.raise_for_status()
            logger.info("Ticket reply sent id=%s", ticket_id)
            return True

        except httpx.HTTPStatusError as e:
            logger.error("Ticket reply error %s: %s", e.response.status_code, e.response.text[:250])
            return False
        except httpx.RequestError as e:
            logger.error("Network error ticket reply id=%s: %s", ticket_id, e)
            return False

    async def send_note(
            self,
            conversation_id: str,
            text: str,
            admin_id: Optional[str] = None,
    ) -> bool:
        return await self._add_conversation_part(
            conversation_id,
            message_type="note",
            body=text,
            admin_id=admin_id,
        )

    async def assign_to_admin(
            self,
            conversation_id: str,
            admin_id: Optional[str] = None,
    ) -> bool:
        admin_id = admin_id or settings.INTERCOM_ADMIN_ID

        if not admin_id:
            logger.error("INTERCOM_ADMIN_ID не настроен")
            return False

        return await self._add_conversation_part(
            conversation_id,
            message_type="assignment",
            admin_id=admin_id,
            assignee_id=admin_id,
        )

    async def close_conversation(
            self,
            conversation_id: str,
            admin_id: Optional[str] = None,
            reason: Optional[str] = None,
    ) -> bool:
        return await self._add_conversation_part(
            conversation_id,
            message_type="close",
            body=reason or "Closed automatically",
            admin_id=admin_id,
        )

    async def open_conversation(
            self,
            conversation_id: str,
            admin_id: Optional[str] = None,
    ) -> bool:
        return await self._add_conversation_part(
            conversation_id,
            message_type="open",
            admin_id=admin_id,
        )
