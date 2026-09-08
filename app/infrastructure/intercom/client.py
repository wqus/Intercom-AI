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


    async def reply_exists(self, entity_id: str, text: str, *, after_epoch: int | None = None) -> bool:
        """Reconcile an ambiguous conversation reply after an HTTP 5xx.

        Intercom's reply endpoint documents POST success/failure semantics but does
        not document an idempotency-key header for this create-message operation. A
        5xx therefore cannot safely be retried blindly. This method checks the
        conversation's persisted parts and only confirms delivery when the exact
        admin-authored comment is visible there.
        """
        if settings.DRY_RUN:
            return True
        if not settings.INTERCOM_TOKEN or not settings.INTERCOM_ADMIN_ID:
            raise RuntimeError("Intercom credentials are not configured")
        response = await self.client.get(
            f"{BASE}/conversations/{entity_id}",
            params={"display_as": "plaintext"},
        )
        response.raise_for_status()
        data = response.json() or {}
        parts = ((data.get("conversation_parts") or {}).get("conversation_parts") or [])
        expected = str(text or "").strip()
        for part in reversed(parts):
            if not isinstance(part, dict):
                continue
            author = part.get("author") or {}
            if str(author.get("id") or "") != str(settings.INTERCOM_ADMIN_ID):
                continue
            if str(author.get("type") or "") not in {"admin", "bot"}:
                continue
            if str(part.get("part_type") or "") not in {"comment", "reply"}:
                continue
            if str(part.get("body") or "").strip() != expected:
                continue
            if after_epoch is not None:
                try:
                    if int(part.get("created_at") or 0) < int(after_epoch):
                        continue
                except (TypeError, ValueError):
                    continue
            return True
        return False

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

