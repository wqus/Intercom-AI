from __future__ import annotations
import logging
import httpx
from app.core.config import settings

logger = logging.getLogger("intercom")
BASE = "https://api.intercom.io"

class IntercomGateway:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def send_reply(self, entity_id: str, channel: str, text: str) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] reply entity=%s channel=%s", entity_id, channel)
            return True
        admin_id = settings.INTERCOM_ADMIN_ID
        if not settings.INTERCOM_TOKEN or not admin_id:
            raise RuntimeError("Intercom credentials are not configured")
        if channel == "ticket":
            url = f"{BASE}/tickets/{entity_id}/reply"
        else:
            url = f"{BASE}/conversations/{entity_id}/reply"
        response = await self.client.post(url, json={
            "message_type": "comment", "type": "admin",
            "admin_id": admin_id, "body": text,
        })
        response.raise_for_status()
        return True

    async def add_escalation_tag(self, entity_id: str, channel: str) -> None:
        tag_id = settings.INTERCOM_ESCALATION_TAG_ID
        if settings.DRY_RUN or not tag_id:
            return
        admin_id = settings.INTERCOM_ADMIN_ID
        path = "tickets" if channel == "ticket" else "conversations"
        try:
            response = await self.client.post(
                f"{BASE}/{path}/{entity_id}/tags",
                json={"id": tag_id, "admin_id": admin_id},
            )
            if not response.is_success:
                logger.warning("Escalation tag failed: %s", response.text[:300])
        except httpx.HTTPError as exc:
            logger.warning("Escalation tag request failed: %s", exc)

    async def assign(self, entity_id: str, channel: str) -> None:
        if settings.DRY_RUN or not settings.INTERCOM_ADMIN_ID:
            return
        admin_id=settings.INTERCOM_ADMIN_ID
        if channel=="ticket":
            response=await self.client.put(
                f"{BASE}/tickets/{entity_id}",
                json={"admin_id":admin_id,"assignee_id":admin_id},
            )
        else:
            response=await self.client.post(
                f"{BASE}/conversations/{entity_id}/parts",
                json={"message_type":"assignment","type":"admin","admin_id":admin_id,"assignee_id":admin_id},
            )
        response.raise_for_status()

