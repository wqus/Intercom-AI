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
    needs_clarification: bool = False  # NEW
    clarification_kb_id: str | None = None  # NEW


def build_user_message(question: str, relevant_entries: list[dict]) -> str:
    entries_text = "\n\n".join(
        f"[id={entry['id']}] [{entry['category']}]\n"
        f"requires_clarification={entry.get('requires_clarification', False)}\n"
        f"{entry.get('clarification_question') or entry['template_answer']}"
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
                message_item = next(
                    item for item in response.output if item.type == "message"
                )
                text = message_item.content[0].text
                data = json.loads(text)
            except Exception:
                logger.error("Bad LLM structure: %s", response)
                raise

            is_noise = data.get("is_noise", False)
            can_answer = data.get("can_answer", False)
            needs_clarification = data.get("needs_clarification", False)
            clarification_kb_id = data.get("clarification_kb_id")

            logger.info(
                "LLM response | is_noise=%s | can_answer=%s | needs_clarification=%s | reason=%s",
                is_noise,
                can_answer,
                needs_clarification,
                data.get("reason"),
            )

            if is_noise:
                return LLMResult(
                    can_answer=False,
                    answer=None,
                    reason=None,
                    is_noise=True,
                )

            # === НОВОЕ: LLM определила что нужен clarification-сценарий ===
            if needs_clarification and clarification_kb_id:
                return LLMResult(
                    can_answer=False,
                    answer=None,
                    reason=None,
                    is_noise=False,
                    needs_clarification=True,
                    clarification_kb_id=clarification_kb_id,
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