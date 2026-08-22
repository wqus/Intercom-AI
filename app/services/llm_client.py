from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.services.knowledge_base import OPERATOR_HINT

logger = logging.getLogger("llm_client")


@dataclass(slots=True)
class LLMResult:
    can_answer: bool
    answer: str | None
    reason: str | None
    is_noise: bool = False
    needs_clarification: bool = False
    clarification_kb_id: str | None = None


def _entry_for_llm(entry: dict[str, Any]) -> dict[str, Any]:
    """Pass useful intent evidence to the LLM, not only the final template."""
    allowed = (
        "id",
        "category",
        "family",
        "definition",
        "scope",
        "description",
        "positive_examples",
        "match_phrases",
        "negative_phrases",
        "required_phrases",
        "constraints",
        "when_to_use",
        "when_not_to_use",
        "confused_with",
        "requires_clarification",
        "clarification_question",
        "template_answer",
    )
    result = {key: entry[key] for key in allowed if key in entry}
    # Internal scoring fields are useful to the caller but should not dominate
    # the semantic evidence shown to the LLM.
    for key in (
        "_embedding_score",
        "_semantic_score",
        "_bm25_score",
        "_rrf_score",
        "_rerank_score",
        "_final_score",
        "_best_match_phrase",
        "_positive_hits",
        "_negative_hits",
    ):
        if key in entry:
            result[key] = entry[key]
    return result


def build_user_message(question: str, relevant_entries: list[dict]) -> str:
    payload = {
        "task": "Выберите правильный intent для вопроса клиента и, только если он достаточно однозначен, сформируйте ответ.",
        "rules": [
            "Разрешено выбирать ТОЛЬКО один из переданных candidate id.",
            "Нельзя придумывать новый id или использовать категорию вне списка кандидатов.",
            "Учитывайте definition/scope/when_to_use/when_not_to_use и negative_phrases сильнее поверхностного сходства слов.",
            "Если два кандидата близки и нельзя надёжно выбрать один, верните needs_clarification=true и clarification_kb_id одного из кандидатов только если это действительно наиболее подходящий вариант.",
            "Если ни один кандидат не соответствует вопросу, верните is_noise=false, can_answer=false, needs_clarification=false.",
            "Если сообщение является мусором/набором бессмысленных символов, верните is_noise=true.",
            "Ответ должен опираться только на информацию переданных кандидатов и не выдумывать правила бизнеса.",
        ],
        "question": question,
        "candidates": [_entry_for_llm(entry) for entry in relevant_entries],
        "output_schema": {
            "is_noise": "boolean",
            "can_answer": "boolean",
            "needs_clarification": "boolean",
            "clarification_kb_id": "string|null",
            "reason": "string|null",
            "answer": "string|null",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class YandexLLMClient:
    def __init__(self, api_key: str, folder_id: str, prompt_id: str) -> None:
        self.prompt_id = prompt_id
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            default_headers={"OpenAI-Project": folder_id},
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
                input=user_message,
            )

            message_item = next(
                item for item in response.output if item.type == "message"
            )
            text = message_item.content[0].text.strip()

            # Be tolerant of accidental markdown fences around JSON.
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()

            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("LLM output must be a JSON object")

            is_noise = bool(data.get("is_noise", False))
            can_answer = bool(data.get("can_answer", False))
            needs_clarification = bool(data.get("needs_clarification", False))
            clarification_kb_id = data.get("clarification_kb_id")
            if clarification_kb_id is not None:
                clarification_kb_id = str(clarification_kb_id).strip() or None

            answer = data.get("answer")
            answer = answer.strip() if isinstance(answer, str) and answer.strip() else None
            reason = data.get("reason")
            reason = reason.strip() if isinstance(reason, str) and reason.strip() else None

            logger.info(
                "LLM response | noise=%s | can_answer=%s | needs_clarification=%s | clarification_kb_id=%s | reason=%s",
                is_noise,
                can_answer,
                needs_clarification,
                clarification_kb_id,
                reason,
            )

            if is_noise:
                return LLMResult(False, None, reason, is_noise=True)

            # Never let an answer without answer text through.
            if can_answer and not answer:
                can_answer = False
                reason = reason or "missing_answer"

            if needs_clarification and clarification_kb_id:
                return LLMResult(
                    can_answer=False,
                    answer=None,
                    reason=reason,
                    is_noise=False,
                    needs_clarification=True,
                    clarification_kb_id=clarification_kb_id,
                )

            return LLMResult(
                can_answer=can_answer,
                answer=(answer + OPERATOR_HINT) if can_answer and answer else None,
                reason=reason,
                is_noise=False,
            )

        except Exception as exc:
            logger.exception("LLM call/parse failed: %s", exc)
            return LLMResult(
                can_answer=False,
                answer=None,
                reason="llm_unavailable_or_invalid_output",
                is_noise=False,
            )
