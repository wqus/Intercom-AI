import asyncio
import hashlib
import hmac
import json
import logging
import re
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends
from httpx import Response

from app.services.intercom_client import IntercomClient
from app.services.escalation import escalate_to_operator
from app.core.config import settings
from app.core.dependencies import get_llm_client, get_knowledge_base
from app.services.knowledge_base import KnowledgeBase
from app.services.llm_client import YandexLLMClient
from app.services.ticket_handler import handle_new_ticket
from app.database.session import SessionLocal

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


def has_conversation_attachments(payload: dict) -> tuple[bool, str | None]:
    try:
        item = payload["data"]["item"]

        question = None
        result = extract_conversation_data(payload)
        if result:
            _, question = result

        # Проверяем вложения в source
        source_attachments = item.get("source", {}).get("attachments")
        if source_attachments and len(source_attachments) > 0:
            return True, question

        # Проверяем вложения в conversation_parts
        parts = item.get("conversation_parts", {}).get("conversation_parts", [])
        for part in parts:
            attachments = part.get("attachments")
            if attachments and len(attachments) > 0:
                return True, question

        return False, question

    except Exception as e:
        logger.error("Error checking conversation attachments: %s", e)
        return False, None


def has_ticket_attachments(payload: dict) -> tuple[bool, str | None]:
    try:
        item = payload["data"]["item"]

        question = None
        result = extract_ticket_data(payload)
        if result:
            _, question = result

        # Проверяем вложения в ticket_part
        ticket_part = item.get("ticket_part", {})
        attachments = ticket_part.get("attachments")
        if attachments and len(attachments) > 0:
            return True, question

        # Проверяем вложения в ticket_parts
        ticket = item.get("ticket", {})
        parts = ticket.get("ticket_parts", {}).get("ticket_parts", [])

        for part in reversed(parts):
            if part.get("author", {}).get("type") != "user":
                continue

            attachments = part.get("attachments")
            if attachments and len(attachments) > 0:
                return True, question

            return False, question

        return False, question

    except Exception as e:
        logger.error("Error checking ticket attachments: %s", e)
        return False, None

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

def extract_user_and_message(payload: dict) -> Optional[Tuple[str, str, str, str]]:
    """
    Извлекает из вебхука Intercom:
      - updated_at (временная метка Unix)
      - user_name (имя пользователя)
      - user_email (email пользователя)
      - last_message (текст последнего сообщения пользователя)
    Возвращает кортеж (updated_at, user_name, user_email, last_message) или None.
    Для ticket.created имя и email будут пустыми строками.
    """
    try:
        data = payload.get("data", {})
        item = data.get("item", {})
        topic = payload.get("topic", "")

        updated_at = str(item.get("updated_at", ""))

        user_name = ""
        user_email = ""
        last_message = ""

        # Conversation
        if topic in ("conversation.user.created", "conversation.user.replied"):
            source = item.get("source", {})
            author = source.get("author", {})
            user_name = author.get("name", "")
            user_email = author.get("email", "")

            parts = item.get("conversation_parts", {}).get("conversation_parts", [])
            user_parts = [p for p in parts if p.get("author", {}).get("type") == "user"]
            if user_parts:
                last_part = user_parts[-1]
                last_message = last_part.get("body", "")
            else:
                last_message = source.get("body", "")

        #  Ticket created
        elif topic == "ticket.created":
            attrs = item.get("ticket_attributes", {})
            last_message = attrs.get("_default_description_", "")

        # Ticket contact replied
        elif topic == "ticket.contact.replied":
            ticket_part = item.get("ticket_part", {})
            author = ticket_part.get("author", {})
            user_name = author.get("name", "")
            user_email = author.get("email", "")
            last_message = ticket_part.get("body", "")

        else:
            return None

        if last_message:
            last_message = re.sub(r"<[^>]+>", "", last_message).strip()

        if not last_message:
            return None

        return updated_at, user_name, user_email, last_message

    except Exception as e:
        logger.error("Failed to extract user and message: %s", e)
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

async def process_ticket(
    conversation_id: str,
    question: str,
    llm_client: YandexLLMClient,
    knowledge_base: KnowledgeBase,
    is_ticket: bool = False,
):
    async with SessionLocal() as session:
        await handle_new_ticket(
            conversation_id=conversation_id,
            question=question,
            llm_client=llm_client,
            knowledge_base=knowledge_base,
            session=session,
            is_ticket=is_ticket,
        )


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

    logger.info("=" * 80)
    logger.info("RAW WEBHOOK RECEIVED:")
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("=" * 80)

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

    user_info = extract_user_and_message(payload)
    if user_info:
        updated_at, user_name, user_email, _ = user_info
    else:
        updated_at = user_name = user_email = ""

    if topic in TICKET_TOPICS:
        has_attachments, question = has_ticket_attachments(payload)

        if has_attachments:
            logger.info(
                "Ticket %s contains attachments. Escalating immediately.",
                item_id,
            )

            intercom_client = IntercomClient()

            try:
                await escalate_to_operator(
                    conversation_id=item_id,
                    question=question or "Не удалось извлечь текст обращения",
                    reason="Обращение содержит вложения",
                    is_ticket=True,
                    intercom_client=intercom_client,
                )
            finally:
                await intercom_client.close()

            return {
                "status": "escalated",
                "reason": "attachments",
            }

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
            process_ticket(
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

    has_attachments, question = has_conversation_attachments(payload)

    if has_attachments:
        logger.info("Item %s contains attachments. Escalating immediately.", item_id)

        intercom_client = IntercomClient()

        try:
            # Эскалируем оператору
            await escalate_to_operator(
                conversation_id=item_id,
                question=question or "Не удалось извлечь текст обращения",
                reason="Обращение содержит вложения",
                is_ticket=False,
                intercom_client=intercom_client,
            )
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
        process_ticket(
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
