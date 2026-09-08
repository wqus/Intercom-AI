from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.core.config import settings
from app.infrastructure.persistence.outbox import OutboundRepository
from app.infrastructure.intercom.client import IntercomGateway
from app.infrastructure.telegram.notifier import TelegramNotifier

logger = logging.getLogger("outbox_worker")


class OutboxWorker:
    """Deliver transactional-outbox side effects independently from event processing."""

    def __init__(self, intercom: IntercomGateway, telegram: TelegramNotifier):
        self.intercom = intercom
        self.telegram = telegram
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="intercom-outbox-worker")
        logger.info("Outbox worker started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _deliver(self, action) -> tuple[str, str | None]:
        payload = action.payload or {}
        action_type = action.action_type
        if action_type == "intercom_reply":
            await self.intercom.send_reply(
                str(payload["entity_id"]),
                str(payload["channel"]),
                str(payload["text"]),
            )
            return "sent", None

        if action_type == "intercom_escalation_tag":
            await self.intercom.add_escalation_tag(
                str(payload["entity_id"]),
                str(payload["channel"]),
            )
            return "sent", None

        if action_type == "telegram_escalation":
            configured = self.telegram.is_configured()
            if not configured:
                return "skipped", "Telegram is not configured"
            ok = await self.telegram.notify_escalation(**payload)
            if ok:
                return "sent", None
            return "retry", "Telegram delivery returned False"

        return "skipped", f"Unknown outbound action type: {action_type}"

    async def _handle_http_status_error(self, action, repo, exc: httpx.HTTPStatusError) -> None:
        """Apply provider-specific failure semantics without blind duplicate sends."""
        status_code = exc.response.status_code if exc.response is not None else 0
        if 500 <= status_code <= 599:
            # Intercom's reply POST does not document an idempotency key. A 5xx can
            # therefore be ambiguous. Reconcile conversation replies before deciding
            # whether delivery happened; otherwise quarantine as unknown.
            if action.action_type == "intercom_reply" and action.payload.get("channel") == "conversation":
                try:
                    after_epoch = None
                    created_at = getattr(action, "created_at", None)
                    if created_at is not None:
                        after_epoch = int(created_at.timestamp()) - 5
                    confirmed = await self.intercom.reply_exists(
                        str(action.payload["entity_id"]),
                        str(action.payload["text"]),
                        after_epoch=after_epoch,
                    )
                except Exception as reconcile_exc:
                    logger.error(
                        "Outbound reconciliation failed action=%s: %s",
                        action.id, reconcile_exc,
                    )
                    confirmed = False
                if confirmed:
                    await repo.mark_sent(action)
                else:
                    await repo.mark_unknown(
                        action,
                        f"Intercom {status_code} after reply POST; delivery requires reconciliation",
                    )
            else:
                await repo.mark_unknown(
                    action,
                    f"Provider {status_code} after outbound write; delivery requires reconciliation",
                )
            return
        if status_code == 429:
            # Rate limiting is retryable and is distinct from ambiguous server-side
            # failures of a create operation.
            await repo.mark_retry(action, str(exc))
            return
        await repo.mark_skipped(action, str(exc))

    async def _run(self) -> None:
        # Import the database session lazily so pure delivery tests and tooling do
        # not require a live SQL driver at module-import time.
        from app.database.session import SessionLocal

        while not self._stop.is_set():
            processed = False
            try:
                async with SessionLocal() as session:
                    repo = OutboundRepository(session)
                    await repo.mark_stale_unknown()
                    action = await repo.claim_one()
                    if action:
                        processed = True
                        try:
                            status, error = await self._deliver(action)
                            if status == "sent":
                                await repo.mark_sent(action)
                            elif status == "skipped":
                                await repo.mark_skipped(action, error or "skipped")
                            else:
                                await repo.mark_retry(action, error or "delivery failed")
                        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                            # A network timeout is delivery-ambiguous: the provider may
                            # already have accepted the write. Do not blindly retry a
                            # customer-facing action and risk a duplicate message.
                            logger.error(
                                "Outbound delivery uncertain action=%s type=%s: %s",
                                action.id,
                                action.action_type,
                                exc,
                            )
                            await repo.mark_unknown(action, str(exc))
                        except httpx.HTTPStatusError as exc:
                            await self._handle_http_status_error(action, repo, exc)
                        except Exception as exc:
                            logger.exception("Outbound delivery failed action=%s", action.id)
                            await repo.mark_retry(action, str(exc))
            except Exception:
                logger.exception("Outbox worker loop failure")
            if not processed:
                await asyncio.sleep(settings.WORKER_POLL_SECONDS)
