from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.persistence.models import OutboundAction
from app.domain.enums import EventStatus


class OutboundRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue(
        self,
        *,
        action_key: str,
        action_type: str,
        entity_id: str,
        channel: str,
        payload: dict,
        event_id: int | None = None,
    ) -> bool:
        stmt = (
            pg_insert(OutboundAction)
            .values(
                action_key=action_key,
                action_type=action_type,
                entity_id=entity_id,
                channel=channel,
                payload=payload,
                event_id=event_id,
            )
            .on_conflict_do_nothing(index_elements=[OutboundAction.action_key])
        )
        result = await self.session.execute(stmt)
        return result.rowcount == 1

    async def claim_one(self) -> OutboundAction | None:
        now = datetime.now(timezone.utc)
        # A stale `processing` action is delivery-ambiguous: the provider may
        # already have accepted the write before this worker crashed. Reclaiming
        # it automatically can therefore duplicate a customer-facing reply.
        #
        # Stale actions are quarantined by `mark_stale_unknown()` instead of
        # being retried. Only explicitly pending actions are claimable.
        stmt = (
            select(OutboundAction)
            .where(
                (OutboundAction.status == "pending")
                & (OutboundAction.available_at <= now)
            )
            .order_by(OutboundAction.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        result = await self.session.execute(stmt)
        action = result.scalar_one_or_none()
        if action:
            action.status = "processing"
            action.attempts += 1
            action.locked_at = now
            await self.session.commit()
        return action

    async def mark_stale_unknown(self, *, older_than_minutes: int = 15) -> int:
        """Quarantine abandoned processing actions instead of replaying them.

        A worker crash after a provider accepted an external write but before
        `mark_sent()` makes delivery ambiguous. Keeping the action `processing`
        forever is operationally poor, but automatically replaying it is unsafe.
        Marking it `unknown` preserves the event for manual reconciliation without
        risking a duplicate customer-facing message.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=int(older_than_minutes))
        result = await self.session.execute(
            update(OutboundAction)
            .where(
                (OutboundAction.status == "processing")
                & (OutboundAction.locked_at.is_not(None))
                & (OutboundAction.locked_at <= cutoff)
            )
            .values(
                status="unknown",
                locked_at=None,
                last_error="stale processing action quarantined; delivery requires reconciliation",
            )
        )
        await self.session.commit()
        return int(result.rowcount or 0)

    async def mark_sent(self, action: OutboundAction, *, commit: bool = True) -> None:
        action.status = "sent"
        action.sent_at = datetime.now(timezone.utc)
        action.locked_at = None
        action.last_error = None
        if commit:
            await self.session.commit()

    async def mark_skipped(self, action: OutboundAction, reason: str, *, commit: bool = True) -> None:
        action.status = "skipped"
        action.locked_at = None
        action.last_error = reason[:4000]
        action.sent_at = datetime.now(timezone.utc)
        if commit:
            await self.session.commit()

    async def mark_retry(self, action: OutboundAction, error: str) -> None:
        action.status = "pending" if action.attempts < 5 else "failed"
        action.locked_at = None
        action.last_error = error[:4000]
        if action.status == "pending":
            delay = min(120, 2 ** max(0, action.attempts - 1))
            action.available_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        await self.session.commit()

    async def mark_failed(self, action: OutboundAction, error: str, *, commit: bool = True) -> None:
        """Persist a terminal delivery failure without pretending it was skipped."""
        action.status = "failed"
        action.locked_at = None
        action.last_error = error[:4000]
        if commit:
            await self.session.commit()

    async def mark_unknown(self, action: OutboundAction, error: str, *, commit: bool = True) -> None:
        # For timeout/network ambiguity we deliberately do NOT auto-retry an external
        # write: the provider may already have accepted it. Manual reconciliation is
        # safer than creating a duplicate customer-facing reply.
        action.status = "unknown"
        action.locked_at = None
        action.last_error = error[:4000]
        if commit:
            await self.session.commit()
