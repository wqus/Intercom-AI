import json
import logging

from dataclasses import dataclass
from app.services.knowledge_base import OPERATOR_HINT
from openai import AsyncOpenAI


logger = logging.getLogger("llm_client")


@dataclass(slots=True)
class LLMResult:
    can_answer: bool
    answer: str | None
    reason: str | None
    is_noise: bool = False


def build_user_message(question: str, relevant_entries: list[dict]) -> str:
    entries_text = "\n\n".join(
        f"[{entry['category']}]\n{entry['template_answer']}"
        for entry in relevant_entries
    )
    return (
        f"База знаний:\n{entries_text}\n\n"
        f"Вопрос клиента: {question}"
    )


class YandexLLMClient:

    def __init__(self, api_key: str, folder_id: str, prompt_id: str) -> None:
        self.prompt_id = prompt_id

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={
                "OpenAI-Project": folder_id,
            }
        )

    async def classify(
        self,
        question: str,
        relevant_entries: list[dict],
    ) -> LLMResult:
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

            is_noise = data.get("is_noise", False)
            can_answer = data.get("can_answer", False)

            logger.info(
                "LLM response | is_noise=%s | can_answer=%s | reason=%s",
                is_noise,
                can_answer,
                data.get("reason"),
            )

            # === НОВОЕ: если это шум — не трогаем answer/reason по обычной логике ===
            if is_noise:
                return LLMResult(
                    can_answer=False,
                    answer=None,
                    reason=None,
                    is_noise=True,
                )

            return LLMResult(
                can_answer=can_answer,
                answer=(data.get("answer") + OPERATOR_HINT if can_answer else None),
                reason=data.get("reason") if not can_answer else None,
                is_noise=False,
            )

        except Exception as e:
            logger.error("LLM call failed: %s — escalating", e)
            return LLMResult(
                can_answer=False,
                answer=None,
                reason="llm_unavailable",
                is_noise=False,
            )