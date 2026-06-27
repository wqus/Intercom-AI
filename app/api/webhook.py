import asyncio
import hashlib
import hmac
import logging
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends
from httpx import Response

from app.core.config import settings
from app.core.dependencies import get_llm_client, get_knowledge_base
from app.services.knowledge_base import KnowledgeBase
from app.services.llm_client import YandexLLMClient
from app.services.ticket_handler import handle_new_ticket

logger = logging.getLogger("webhook")

router = APIRouter()

# защита от дублирования webhook (in-memory)
processed_events = set()


def verify_intercom_signature(body: bytes, signature: str) -> bool:
    if not settings.INTERCOM_CLIENT_SECRET or not signature:
        return False

    expected = "sha1=" + hmac.new(
        settings.INTERCOM_CLIENT_SECRET.encode("utf-8"),
        body,
        hashlib.sha1,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def extract_conversation_data(payload: dict) -> tuple[str, str] | None:
    try:
        item = payload["data"]["item"]
        conversation_id = str(item["id"])

        # Основной источник
        body = item.get("source", {}).get("body", "")

        # Fallback — последнее сообщение пользователя
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
) -> dict|Response:
    # HEAD запрос — Intercom проверяет доступность endpoint
    if request.method == "HEAD":
        return Response(status_code=200)

    body_bytes = await request.body()
    payload = await request.json()

    # === 1. Верификация подписи ===
    signature = request.headers.get("X-Hub-Signature", "")
    if settings.INTERCOM_CLIENT_SECRET:
        if not verify_intercom_signature(body_bytes, signature):
            logger.warning("Invalid Intercom webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logger.warning("INTERCOM_CLIENT_SECRET not set — signature check skipped (DEV)")

    # === 2. Защита от дублирования ===
    event_id = payload.get("id")
    conversation_id = str(payload.get("data", {}).get("item", {}).get("id", ""))

    if event_id and conversation_id:
        event_key = f"{conversation_id}:{event_id}"
        if event_key in processed_events:
            logger.info("Duplicate webhook skipped | conv=%s", conversation_id)
            return {"status": "duplicate"}

        processed_events.add(event_key)

        # Очищаем старые записи, чтобы память не росла
        if len(processed_events) > 2000:
            processed_events.clear()

    # === 3. Фильтрация событий ===
    topic = payload.get("topic", "")
    if topic not in ("conversation.user.created", "conversation.user.replied"):
        return {"status": "ignored", "topic": topic}

    # === 4. Извлечение данных ===
    result = extract_conversation_data(payload)
    if not result:
        logger.warning("Could not extract conversation data")
        return {"status": "skipped", "reason": "no_question"}

    conversation_id, question = result

    logger.info(
        "Webhook accepted | conv=%s | question=%s",
        conversation_id,
        question[:120]
    )

    # === 5. Асинхронная обработка ===
    asyncio.create_task(
        handle_new_ticket(
            conversation_id=conversation_id,
            question=question,
            llm_client=llm_client,
            knowledge_base=knowledge_base,
        )
    )

    return {"status": "accepted"}