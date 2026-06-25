"""
LLM клиент для Yandex AI Studio — Responses API с агентом.

Архитектура:
- Промпт, температура, max_tokens настроены в агенте на сайте AI Studio
- Код только передаёт вопрос + базу знаний и читает ответ
- JSON Schema гарантирует валидный структурированный ответ без ручного парсинга
- Один запрос к Pro модели, без двухуровневого fallback (убрали по требованию)
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger("llm_client")


# ==========================================================
# JSON SCHEMA — гарантирует структуру ответа
# Yandex поддерживает response_format json_schema
# модель ОБЯЗАНА вернуть этот формат, парсинг не нужен
# ==========================================================

# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class LLMResult:
    can_answer: bool
    answer: str | None       # ответ клиенту (если can_answer=True)
    reason: str | None       # причина эскалации (если can_answer=False)


# ==========================================================
# USER MESSAGE
# Системный промпт живёт в агенте на сайте AI Studio.
# Сюда передаём только данные: вопрос + релевантные записи базы знаний.
# ==========================================================

def build_user_message(question: str, relevant_entries: list[dict]) -> str:
    entries_text = "\n\n".join(
        f"[{entry['category']}]\n{entry['template_answer']}"
        for entry in relevant_entries
    )
    return (
        f"База знаний:\n{entries_text}\n\n"
        f"Вопрос клиента: {question}"
    )


# ==========================================================
# CLIENT
# ==========================================================

class YandexLLMClient:

    def __init__(self, api_key: str, folder_id: str, prompt_id: str) -> None:
        self.prompt_id = prompt_id

        # AsyncOpenAI — асинхронная версия для FastAPI
        # base_url и project из актуальной документации Yandex AI Studio
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={
                # folder_id передаётся через project согласно документации
                "OpenAI-Project": folder_id,
            }
        )

    async def classify(
        self,
        question: str,
        relevant_entries: list[dict],
    ) -> LLMResult:
        """
        Отправляет вопрос + релевантные записи базы знаний агенту.
        Промпт, температура, max_tokens настроены в агенте на сайте.
        Возвращает структурированный LLMResult.
        """
        user_message = build_user_message(question, relevant_entries)

        try:
            response = await self._client.responses.create(
                prompt={"id": self.prompt_id},
                input=user_message
            )

            try:
                text = response.output[0].content[0].text
                data = json.loads(text)
            except Exception:
                logger.error("Bad LLM structure: %s", response)
                raise

            can_answer = data.get("can_answer", False)

            logger.info(
                "LLM response | can_answer=%s | reason=%s",
                can_answer,
                data.get("reason"),
            )

            return LLMResult(
                can_answer=can_answer,
                answer=data.get("answer") if can_answer else None,
                reason=data.get("reason") if not can_answer else None,
            )

        except Exception as e:
            logger.error("LLM call failed: %s — escalating", e)
            return LLMResult(
                can_answer=False,
                answer=None,
                reason="llm_unavailable",
            )