import asyncio
import hashlib
import hmac
import json
import logging
import re
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


def has_conversation_attachments(payload: dict) -> bool:
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


def has_ticket_attachments(payload: dict) -> bool:
    try:
        item = payload["data"]["item"]

        ticket_part = item.get("ticket_part", {})
        if ticket_part.get("attachments"):
            return True

        ticket = item.get("ticket", {})
        parts = ticket.get("ticket_parts", {}).get("ticket_parts", [])

        for part in reversed(parts):
            author_type = part.get("author", {}).get("type")

            if author_type == "user":
                if part.get("attachments"):
                    return True
                return False

        return False

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def extract_conversation_data(payload: dict) -> tuple[str, str] | None:
    try:
        item = payload["data"]["item"]
        conversation_id = str(item["id"])
        topic = payload.get("topic", "")

        body = ""

        if topic == "conversation.user.replied":
            parts = item.get("conversation_parts", {}).get("conversation_parts", [])
            for part in reversed(parts):
                if part.get("author", {}).get("type") == "user":
                    body = part.get("body", "")
                    break

        if not body:
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


def extract_ticket_data(payload: dict) -> tuple[str, str] | None:
    try:
        item = payload["data"]["item"]
        topic = payload.get("topic", "")

        if topic == "ticket.contact.replied":
            ticket_id = str(item["ticket"]["id"])
            body = item.get("ticket_part", {}).get("body", "")

            if not body:
                return None

            question = re.sub(r"<[^>]+>", "", body).strip()
            return ticket_id, question

        ticket_id = str(item["id"])
        attrs = item.get("ticket_attributes", {})
        question = attrs.get("_default_description_", "")

        if not question:
            return None

        question = re.sub(r"<[^>]+>", "", question).strip()
        return ticket_id, question

    except Exception as e:
        logger.warning("Failed to extract ticket data: %s", e)
        return None

def get_item_id(payload: dict) -> str | None:
    try:
        data = payload.get("data", {})
        item = data.get("item", {})

        if item_id := item.get("id"):
            return str(item_id)

        if ticket := item.get("ticket"):
            if ticket_id := ticket.get("id") or ticket.get("ticket_id"):
                return str(ticket_id)

        if ticket_part := item.get("ticket_part"):
            if ticket := ticket_part.get("ticket"):
                if ticket_id := ticket.get("id") or ticket.get("ticket_id"):
                    return str(ticket_id)

        return None
    except Exception as e:
        logger.warning("Failed to get item_id: %s", e)
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

    # Защита от дублей
    item_id = get_item_id(payload)
    event_id = payload.get("id")

    if event_id and item_id:
        event_key = f"{item_id}:{event_id}"

        if event_key in processed_events:
            logger.info(
                "Duplicate webhook skipped | id=%s",
                item_id,
            )
            return {
                "status": "duplicate",
            }

        processed_events.add(event_key)

        if len(processed_events) > 2000:
            processed_events.clear()

    if topic in TICKET_TOPICS:
        result = None
        if topic in TICKET_TOPICS:
            if has_ticket_attachments(payload):
                logger.info("Ticket %s contains attachments. Escalating immediately.", item_id)
                intercom_client = IntercomClient()
                try:
                    message = "Ваше сообщение принято. Мы обработаем его в ближайшее время и свяжемся с вами."
                    await intercom_client.send_ticket_reply(item_id, message)
                finally:
                    await intercom_client.close()
                return {"status": "escalated", "reason": "attachments"}

            result = extract_ticket_data(payload)

        if not result:
            logger.warning("Could not extract ticket data")

            return {
                "status": "skipped",
                "reason": "no_question",
            }

        ticket_id, question = result

        logger.info(
            "Ticket accepted | id=%s | question=%s",
            ticket_id,
            question[:120],
        )

        asyncio.create_task(
            handle_new_ticket(
                conversation_id=ticket_id,
                question=question,
                llm_client=llm_client,
                knowledge_base=knowledge_base,
                is_ticket=True,
            )
        )

        return {
            "status": "accepted",
        }

    if topic not in CONVERSATION_TOPICS:
        return {"status": "ignored", "topic": topic}

    item = payload.get("data", {}).get("item", {})

    if item.get("ticket") is not None:
        logger.info(
            "Conversation %s is linked to a ticket — skipping, handled via ticket topic",
            item.get("id")
        )
        return {"status": "skipped", "reason": "linked_to_ticket"}

    if has_conversation_attachments(payload):
        logger.info(
            "Item %s contains attachments. Escalating immediately.",
            item_id,
        )

        intercom_client = IntercomClient()

        try:
            message = "Ваше сообщение принято. Мы обработаем его в ближайшее время и свяжемся с вами."
            await intercom_client.send_conversations_reply(item_id, message)
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
