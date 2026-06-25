
import asyncio
import hashlib
import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

import re
from app.core.config import settings
from app.services.knowledge_base import KnowledgeBase
from app.services.llm_client import YandexLLMClient
from app.services.ticket_handler import handle_new_ticket

logger = logging.getLogger("webhook")

router = APIRouter()


# ВЕРИФИКАЦИЯ ПОДПИСИ
def verify_intercom_signature(body: bytes, signature: str) -> bool:

    if not signature:
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

        # текст первого сообщения от пользователя
        source = item.get("source", {})
        body = source.get("body", "")

        if body:
            # убираем HTML теги
            import re
            question = re.sub(r"<[^>]+>", "", body).strip()
        else:
            # fallback — берём из conversation_parts если source пустой
            parts = item.get("conversation_parts", {}).get("conversation_parts", [])
            user_parts = [p for p in parts if p.get("author", {}).get("type") == "user"]
            if not user_parts:
                return None
            question = re.sub(r"<[^>]+>", "", user_parts[0].get("body", "")).strip()

        if not question:
            return None

        return conversation_id, question

    except (KeyError, TypeError) as e:
        logger.warning("Failed to extract conversation data: %s", e)
        return None


# WEBHOOK ENDPOINT
@router.post("/webhook/intercom")
async def intercom_webhook(
    request: Request,
    llm: YandexLLMClient = Depends(lambda: __import__("app.main", fromlist=["get_llm_client"]).get_llm_client()),
    kb: KnowledgeBase = Depends(lambda: __import__("app.main", fromlist=["get_knowledge_base"]).get_knowledge_base()),
):
    body = await request.body()

    # ШАГ 1: верификация
    signature = request.headers.get("X-Hub-Signature", "")

    # пропускаем верификацию в dev режиме если секрет не задан
    if settings.INTERCOM_CLIENT_SECRET:
        if not verify_intercom_signature(body, signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    else:
        logger.warning("INTERCOM_CLIENT_SECRET not set — skipping signature check")

    # ШАГ 2: парсинг
    payload = await request.json()
    topic = payload.get("topic", "")

    # обрабатываем только новые сообщения от пользователей
    if topic not in ("conversation.user.created", "conversation.user.replied"):
        logger.debug("Ignoring topic: %s", topic)
        return {"status": "ignored", "topic": topic}

    result = extract_conversation_data(payload)

    if result is None:
        logger.warning("Could not extract conversation data from payload")
        return {"status": "skipped", "reason": "no_question"}

    conversation_id, question = result

    logger.info("New ticket | conversation_id=%s | question=%s", conversation_id, question[:100])

    asyncio.create_task(
        handle_new_ticket(
            conversation_id=conversation_id,
            question=question,
            llm_client=llm,
            knowledge_base=kb,
        )
    )

    return {"status": "received"}