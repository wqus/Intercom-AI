import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("intercom_client")

INTERCOM_API_BASE = "https://api.intercom.io"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.INTERCOM_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Intercom рекомендует указывать версию API явно
        "Intercom-Version": "2.11",
    }


async def send_reply(conversation_id: str, text: str) -> bool:
    if settings.DRY_RUN:
        logger.info(
            "[DRY RUN] Would send reply to conversation %s: %s",
            conversation_id,
            text[:100],
        )
        return True

    if not settings.INTERCOM_TOKEN:
        logger.error("INTERCOM_TOKEN not set — cannot send reply")
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{INTERCOM_API_BASE}/conversations/{conversation_id}/reply",
                headers=_headers(),
                json={
                    "type": "admin",
                    "message_type": "comment",
                    "body": text,
                },
            )
            response.raise_for_status()
            logger.info("Reply sent to conversation %s", conversation_id)
            return True

    except httpx.HTTPStatusError as e:
        logger.error(
            "Intercom reply error %s for conversation %s: %s",
            e.response.status_code,
            conversation_id,
            e.response.text[:200],
        )
        return False
    except httpx.RequestError as e:
        logger.error("Intercom request error for conversation %s: %s", conversation_id, e)
        return False


async def assign_to_team(conversation_id: str) -> bool:
    if settings.DRY_RUN:
        logger.info(
            "[DRY RUN] Would assign conversation %s to team %s",
            conversation_id,
            settings.SUPPORT_TEAM_ID,
        )
        return True

    if not settings.INTERCOM_TOKEN:
        logger.error("INTERCOM_TOKEN not set — cannot assign conversation")
        return False

    if not settings.SUPPORT_TEAM_ID:
        logger.error("SUPPORT_TEAM_ID not set — cannot assign conversation")
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.put(
                f"{INTERCOM_API_BASE}/conversations/{conversation_id}",
                headers=_headers(),
                json={
                    "assignment": {
                        "type": "team",
                        "admin_id": settings.SUPPORT_TEAM_ID,
                    }
                },
            )
            response.raise_for_status()
            logger.info(
                "Conversation %s assigned to team %s",
                conversation_id,
                settings.SUPPORT_TEAM_ID,
            )
            return True

    except httpx.HTTPStatusError as e:
        logger.error(
            "Intercom assign error %s for conversation %s: %s",
            e.response.status_code,
            conversation_id,
            e.response.text[:200],
        )
        return False
    except httpx.RequestError as e:
        logger.error("Intercom request error for conversation %s: %s", conversation_id, e)
        return False