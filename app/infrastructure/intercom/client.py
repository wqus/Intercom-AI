from __future__ import annotations
import logging
import time
import httpx
from app.core.config import settings

logger = logging.getLogger("intercom")
BASE = "https://api.intercom.io"


class IntercomGateway:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @staticmethod
    def _role_for_author(author: dict) -> str | None:
        author_type = str((author or {}).get("type") or "").strip().lower()
        if author_type == "user":
            return "user"
        if author_type in {"admin", "bot", "teammate", "assistant"}:
            return "assistant"
        return None

    @staticmethod
    def _timestamp(value) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parts_to_context(
        cls,
        parts: list[dict],
        *,
        current_part_id: str | None = None,
        before_epoch: int | None = None,
        limit: int = 12,
    ) -> list[dict[str, str]]:
        rows: list[tuple[int, int, dict[str, str]]] = []
        for idx, part in enumerate(parts or []):
            if not isinstance(part, dict):
                continue
            pid = str(part.get("id") or part.get("part_id") or "")
            if current_part_id and pid == current_part_id:
                continue
            body = str(part.get("body") or "").strip()
            if not body:
                continue
            part_type = str(part.get("part_type") or "").strip().lower()
            if part_type and part_type not in {"comment", "reply", "message"}:
                continue
            stamp = cls._timestamp(part.get("created_at"))
            if before_epoch is not None and stamp is not None and stamp > before_epoch:
                continue
            role = cls._role_for_author(part.get("author") or {})
            if role is None:
                continue
            rows.append((stamp if stamp is not None else -1, idx, {"role": role, "text": body}))
        rows.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in rows[-limit:]]

    async def send_reply(self, entity_id: str, channel: str, text: str) -> bool:
        if settings.DRY_RUN:
            logger.info("[DRY RUN] reply entity=%s channel=%s", entity_id, channel)
            return True
        admin_id = settings.INTERCOM_ADMIN_ID
        if not settings.INTERCOM_TOKEN or not admin_id:
            raise RuntimeError("Intercom credentials are not configured")
        if channel == "ticket":
            url = f"{BASE}/tickets/{entity_id}/reply"
        elif channel == "conversation":
            url = f"{BASE}/conversations/{entity_id}/reply"
        else:
            raise ValueError(f"Unsupported Intercom channel: {channel!r}")
        request_payload = {
            "message_type": "comment",
            "type": "admin",
            "admin_id": admin_id,
            "body": text,
        }
        started = time.perf_counter()
        logger.info(
            "intercom_reply_start | channel=%s | entity=%s | body_len=%d",
            channel, entity_id, len(str(text or "")),
        )
        response = await self.client.post(url, json=request_payload)
        response.raise_for_status()
        try:
            data = response.json() or {}
            logger.info(
                "intercom_reply_success | channel=%s | entity=%s | status=%s | response_type=%s | response_id=%s | part_type=%s | elapsed_ms=%.1f",
                channel, entity_id, response.status_code,
                data.get("type"), data.get("id"), data.get("part_type"),
                (time.perf_counter() - started) * 1000,
            )
        except ValueError:
            logger.warning(
                "intercom_reply_success_non_json | channel=%s | entity=%s | status=%s | elapsed_ms=%.1f",
                channel, entity_id, response.status_code,
                (time.perf_counter() - started) * 1000,
            )
        return True

    async def get_conversation_context(
        self,
        entity_id: str,
        *,
        current_part_id: str | None = None,
        before_epoch: int | None = None,
        limit: int = 12,
    ) -> list[dict[str, str]]:
        if settings.DRY_RUN:
            return []
        response = await self.client.get(
            f"{BASE}/conversations/{entity_id}",
            params={"display_as": "plaintext"},
            timeout=float(settings.CONTEXT_FETCH_TIMEOUT),
        )
        response.raise_for_status()
        data = response.json() or {}
        parts = ((data.get("conversation_parts") or {}).get("conversation_parts") or [])
        context = self._parts_to_context(
            parts,
            current_part_id=current_part_id,
            before_epoch=before_epoch,
            limit=limit,
        )
        # The conversation `source` often contains the original customer message
        # even when it is not repeated in conversation_parts. Include it when it is
        # clearly customer-authored and not the current triggering message.
        source = data.get("source") or {}
        source_role = self._role_for_author(source.get("author") or {})
        source_body = str(source.get("body") or "").strip()
        source_id = str(source.get("id") or "")
        source_time = self._timestamp(source.get("created_at"))
        if (
            source_role == "user"
            and source_body
            and source_id != str(current_part_id or "")
            and (before_epoch is None or source_time is None or source_time <= before_epoch)
        ):
            context = [{"role": "user", "text": source_body}] + context
            context = context[-limit:]
        return context

    async def get_ticket_context(
        self,
        entity_id: str,
        *,
        current_part_id: str | None = None,
        before_epoch: int | None = None,
        limit: int = 12,
    ) -> list[dict[str, str]]:
        if settings.DRY_RUN:
            return []
        response = await self.client.get(f"{BASE}/tickets/{entity_id}", timeout=float(settings.CONTEXT_FETCH_TIMEOUT))
        response.raise_for_status()
        data = response.json() or {}
        parts = ((data.get("ticket_parts") or {}).get("ticket_parts") or [])
        return self._parts_to_context(
            parts,
            current_part_id=current_part_id,
            before_epoch=before_epoch,
            limit=limit,
        )

    async def _reply_exists_in_parts(
        self,
        parts: list[dict],
        text: str,
        *,
        after_epoch: int | None = None,
    ) -> bool:
        expected = str(text or "").strip()
        for part in reversed(parts or []):
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
                created = self._timestamp(part.get("created_at"))
                if created is not None and created < int(after_epoch):
                    continue
            return True
        return False

    async def reply_exists(
        self,
        entity_id: str,
        text: str,
        *,
        channel: str = "conversation",
        after_epoch: int | None = None,
    ) -> bool:
        """Reconcile an ambiguous admin reply without blind duplicate POSTs."""
        if settings.DRY_RUN:
            return True
        if not settings.INTERCOM_TOKEN or not settings.INTERCOM_ADMIN_ID:
            raise RuntimeError("Intercom credentials are not configured")
        if channel == "ticket":
            response = await self.client.get(f"{BASE}/tickets/{entity_id}")
            response.raise_for_status()
            data = response.json() or {}
            parts = ((data.get("ticket_parts") or {}).get("ticket_parts") or [])
        elif channel == "conversation":
            response = await self.client.get(
                f"{BASE}/conversations/{entity_id}",
                params={"display_as": "plaintext"},
            )
            response.raise_for_status()
            data = response.json() or {}
            parts = ((data.get("conversation_parts") or {}).get("conversation_parts") or [])
        else:
            raise ValueError(f"Unsupported Intercom channel: {channel!r}")
        confirmed = await self._reply_exists_in_parts(parts, text, after_epoch=after_epoch)
        logger.info(
            "intercom_reply_reconciliation | channel=%s | entity=%s | confirmed=%s",
            channel, entity_id, confirmed,
        )
        return confirmed

    async def add_escalation_tag(self, entity_id: str, channel: str) -> bool:
        tag_id = settings.INTERCOM_ESCALATION_TAG_ID
        if settings.DRY_RUN or not tag_id:
            logger.info(
                "intercom_escalation_tag_skipped | entity=%s | channel=%s | configured=%s",
                entity_id, channel, bool(tag_id),
            )
            return True
        if not settings.INTERCOM_ADMIN_ID:
            raise RuntimeError("INTERCOM_ADMIN_ID is required for escalation tagging")
        if channel == "ticket":
            path = "tickets"
        elif channel == "conversation":
            path = "conversations"
        else:
            raise ValueError(f"Unsupported Intercom channel: {channel!r}")
        response = await self.client.post(
            f"{BASE}/{path}/{entity_id}/tags",
            json={"id": tag_id, "admin_id": settings.INTERCOM_ADMIN_ID},
        )
        response.raise_for_status()
        logger.info(
            "intercom_escalation_tag_success | entity=%s | channel=%s | tag_id=%s | status=%s",
            entity_id, channel, tag_id, response.status_code,
        )
        return True
