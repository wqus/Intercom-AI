import asyncio
import hashlib
import hmac
import logging
import re
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends
from httpx import Response
from app.services.intercom_client import IntercomClient

from app.core.config import settings
from app.core.dependencies import get_llm_client, get_knowledge_base
from app.services.knowledge_base import KnowledgeBase
from app.services.llm_client import YandexLLMClient
from app.services.ticket_handler import handle_new_ticket

logger = logging.getLogger("webhook")

router = APIRouter()

processed_events = set()

CONVERSATION_TOPICS = {
    "conversation.user.created",
    "conversation.user.replied",
}

TICKET_TOPICS = {
    "ticket.created",
    "ticket.contact.replied",
}

def verify_intercom_signature(body: bytes, signature: str) -> bool:
    if not settings.INTERCOM_CLIENT_SECRET or not signature:
        return False

    expected = "sha1=" + hmac.new(
        settings.INTERCOM_CLIENT_SECRET.encode("utf-8"),
        body,
        hashlib.sha1,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)

def has_attachments(payload: dict) -> bool:
    try:
        item = payload["data"]["item"]

        if item.get("source", {}).get("attachments"):
            return True

        parts = item.get("conversation_parts", {}).get("conversation_parts", [])
        for part in parts:
            if part.get("attachments"):
                return True

        return False

    except Exception:
        return False

def extract_conversation_data(payload: dict) -> tuple[str, str] | None:
    try:
        item = payload["data"]["item"]
        conversation_id = str(item["id"])

        body = item.get("source", {}).get("body", "")

        if not body:
            parts = item.get("conversation_parts", {}).get("conversation_parts", [])
            for part in reversed(parts):
                if part.get("author", {}).get("type") == "user":
                    body = part.get("body", "")
                    break

        if not body:
            return None

        question = re.sub(r"<[^>]+>", "", body).strip()
        return conversation_id, question

    except Exception as e:
        logger.warning("Failed to extract conversation data: %s", e)
        return None


@router.api_route("/webhook/intercom", methods=["POST", "HEAD"], response_model=None)
async def intercom_webhook(
    request: Request,
    llm_client: YandexLLMClient = Depends(get_llm_client),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
) -> dict | Response:

    if request.method == "HEAD":
        return Response(status_code=200)

    body_bytes = await request.body()
    payload = await request.json()

    # Проверка подписи
    signature = request.headers.get("X-Hub-Signature", "")

    if settings.INTERCOM_CLIENT_SECRET:
        if not verify_intercom_signature(body_bytes, signature):
            logger.warning("Invalid Intercom webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logger.warning(
            "INTERCOM_CLIENT_SECRET not set — signature check skipped (DEV)"
        )

    topic = payload.get("topic", "")

    conversation_topics = {
        "conversation.user.created",
        "conversation.user.replied",
    }

    ticket_topics = {
        "ticket.created",
        "ticket.contact.replied",
    }

    if topic in ticket_topics:
        logger.info(
            "\n========== INTERCOM TICKET ==========\n%s\n=====================================",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

        return {
            "status": "ticket_logged",
            "topic": topic,
        }

    if topic not in conversation_topics:
        return {
            "status": "ignored",
            "topic": topic,
        }

    # Защита от дублей
    event_id = payload.get("id")
    conversation_id = str(payload.get("data", {}).get("item", {}).get("id", ""))

    if event_id and conversation_id:
        event_key = f"{conversation_id}:{event_id}"

        if event_key in processed_events:
            logger.info(
                "Duplicate webhook skipped | conv=%s",
                conversation_id,
            )
            return {
                "status": "duplicate",
            }

        processed_events.add(event_key)

        if len(processed_events) > 2000:
            processed_events.clear()

    # Вложения сразу оператору
    if has_attachments(payload):
        logger.info(
            "Conversation %s contains attachments. Escalating immediately.",
            conversation_id,
        )

        intercom_client = IntercomClient()

        try:
            await intercom_client.assign_to_admin(conversation_id)
        finally:
            await intercom_client.close()

        return {
            "status": "escalated",
            "reason": "attachments",
        }

    result = extract_conversation_data(payload)

    if not result:
        logger.warning("Could not extract conversation data")

        return {
            "status": "skipped",
            "reason": "no_question",
        }

    conversation_id, question = result

    logger.info(
        "Webhook accepted | conv=%s | question=%s",
        conversation_id,
        question[:120],
    )

    asyncio.create_task(
        handle_new_ticket(
            conversation_id=conversation_id,
            question=question,
            llm_client=llm_client,
            knowledge_base=knowledge_base,
        )
    )

    return {
        "status": "accepted",
    }

@router.get("/health")
async def health():
    return {"status": "ok"}