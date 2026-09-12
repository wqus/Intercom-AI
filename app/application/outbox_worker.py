from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.core.config import settings
from app.infrastructure.persistence.repository import ConversationTurnRepository
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

    async def _sync_turn(self, session, action, status: str) -> None:
        if action.action_type != "intercom_reply" or not hasattr(session, "scalar"):
            return
        await ConversationTurnRepository(session).mark_action_status(str(action.action_key), status)
        await session.commit()

    async def _deliver(self, action) -> tuple[str, str | None]:
        payload = action.payload or {}
        action_type = action.action_type
        started = time.perf_counter()
        logger.info(
            "outbox_delivery_start | action=%s | action_type=%s | entity=%s | channel=%s | attempt=%s",
            action.id, action_type, getattr(action, "entity_id", None) or payload.get("entity_id"), getattr(action, "channel", None) or payload.get("channel"), getattr(action, "attempts", None),
        )
        if action_type == "intercom_reply":
            await self.intercom.send_reply(
                str(payload["entity_id"]),
                str(payload["channel"]),
                str(payload["text"]),
            )
            logger.info(
                "outbox_delivery_success | action=%s | action_type=%s | entity=%s | elapsed_ms=%.1f",
                action.id, action_type, getattr(action, "entity_id", None) or payload.get("entity_id"), (time.perf_counter() - started) * 1000,
            )
            return "sent", None

        if action_type == "intercom_escalation_tag":
            await self.intercom.add_escalation_tag(
                str(payload["entity_id"]),
                str(payload["channel"]),
            )
            logger.info(
                "outbox_delivery_success | action=%s | action_type=%s | entity=%s | elapsed_ms=%.1f",
                action.id, action_type, getattr(action, "entity_id", None) or payload.get("entity_id"), (time.perf_counter() - started) * 1000,
            )
            return "sent", None

        if action_type == "telegram_escalation":
            if not self.telegram.is_configured():
                return "skipped", "Telegram is not configured"
            ok = await self.telegram.notify_escalation(**payload)
            if ok:
                logger.info(
                    "outbox_delivery_success | action=%s | action_type=%s | entity=%s | elapsed_ms=%.1f",
                    action.id, action_type, getattr(action, "entity_id", None) or payload.get("entity_id"), (time.perf_counter() - started) * 1000,
                )
                return "sent", None
            return "retry", "Telegram delivery returned False"

        return "skipped", f"Unknown outbound action type: {action_type}"

    async def _handle_http_status_error(self, action, repo, exc: httpx.HTTPStatusError) -> None:
        """Apply provider-specific failure semantics without blind duplicate sends."""
        status_code = exc.response.status_code if exc.response is not None else 0
        action_type = action.action_type
        channel = str((action.payload or {}).get("channel") or action.channel or "")

        if 500 <= status_code <= 599:
            if action_type == "intercom_reply":
                try:
                    created_at = getattr(action, "created_at", None)
                    after_epoch = int(created_at.timestamp()) - 5 if created_at is not None else None
                    confirmed = await self.intercom.reply_exists(
                        str(action.payload["entity_id"]),
                        str(action.payload["text"]),
                        channel=channel,
                        after_epoch=after_epoch,
                    )
                except Exception as reconcile_exc:
                    logger.error(
                        "outbox_reconciliation_failed | action=%s | type=%s | channel=%s | error_type=%s | error=%s",
                        action.id, action_type, channel, type(reconcile_exc).__name__, reconcile_exc,
                    )
                    confirmed = False
                if confirmed:
                    await repo.mark_sent(action)
                    await self._sync_turn(getattr(repo, "session", None), action, "sent")
                else:
                    await self._sync_turn(getattr(repo, "session", None), action, "unknown")
                    await repo.mark_unknown(
                        action,
                        f"Intercom {status_code} after reply POST; delivery requires reconciliation",
                    )
                return

            # Tag updates and operator notifications are not customer-facing message
            # creates. Their 5xx failures are safe to retry and preferable to losing
            # escalation metadata/operator visibility.
            if action_type in {"intercom_escalation_tag", "telegram_escalation"}:
                await repo.mark_retry(action, f"Provider HTTP {status_code}: {exc}")
                return

            await repo.mark_unknown(action, f"Provider {status_code} after outbound write")
            return

        if status_code == 429:
            await repo.mark_retry(action, str(exc))
            return

        if 400 <= status_code <= 499:
            await repo.mark_failed(action, f"Provider returned HTTP {status_code}: {exc}")
            if action.action_type == "intercom_reply":
                await self._sync_turn(getattr(repo, "session", None), action, "failed")
            logger.error(
                "outbox_delivery_permanent_failure | action=%s | type=%s | channel=%s | status=%s",
                action.id, action_type, channel, status_code,
            )
            return

        await repo.mark_retry(action, str(exc))

    async def _run(self) -> None:
        from app.database.session import SessionLocal

        while not self._stop.is_set():
            processed = False
            try:
                async with SessionLocal() as session:
                    repo = OutboundRepository(session)
                    quarantined = await repo.mark_stale_unknown()
                    if quarantined:
                        logger.error("outbox_stale_quarantined | count=%s", quarantined)
                    action = await repo.claim_one()
                    if action:
                        processed = True
                        try:
                            status, error = await self._deliver(action)
                            if status == "sent":
                                await repo.mark_sent(action)
                                await self._sync_turn(session, action, "sent")
                            elif status == "skipped":
                                await repo.mark_skipped(action, error or "skipped")
                                await self._sync_turn(session, action, "skipped")
                            else:
                                await repo.mark_retry(action, error or "delivery failed")
                        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                            # Customer-facing Intercom writes are ambiguous on network
                            # failure. Telegram/tag writes are retryable because losing
                            # the operator signal/metadata is worse than a duplicate.
                            if action.action_type == "intercom_reply":
                                await repo.mark_unknown(action, str(exc))
                                await self._sync_turn(session, action, "unknown")
                                logger.error(
                                    "outbox_delivery_unknown | action=%s | type=%s | entity=%s | error=%s",
                                    action.id, action.action_type, action.entity_id, exc,
                                )
                            else:
                                await repo.mark_retry(action, str(exc))
                                logger.warning(
                                    "outbox_delivery_retry | action=%s | type=%s | entity=%s | error=%s",
                                    action.id, action.action_type, action.entity_id, exc,
                                )
                        except httpx.HTTPStatusError as exc:
                            await self._handle_http_status_error(action, repo, exc)
                        except Exception as exc:
                            # Unknown application errors are ambiguous only for actual
                            # external writes; keep the durable action for manual
                            # reconciliation rather than marking it sent.
                            if action.action_type == "intercom_reply":
                                await repo.mark_unknown(action, str(exc))
                                await self._sync_turn(session, action, "unknown")
                            else:
                                await repo.mark_retry(action, str(exc))
                            logger.exception("Outbox delivery failed action=%s", action.id)
            except Exception:
                logger.exception("Outbox worker loop failure")
            if not processed:
                await asyncio.sleep(settings.WORKER_POLL_SECONDS)
