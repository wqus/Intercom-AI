
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
    selected_kb_id: str | None = None
    is_noise: bool = False
    needs_clarification: bool = False
    clarification_kb_id: str | None = None


def _entry_for_llm(entry: dict[str, Any]) -> dict[str, Any]:
    """Expose the semantic/business evidence needed for a constrained decision."""
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
        "hard_negative_examples",
    )
    result = {key: entry[key] for key in allowed if key in entry}

    # Diagnostic signals are useful for the LLM to understand why the candidate
    # was retrieved, but are explicitly secondary to semantic/business evidence.
    for key in (
        "_embedding_score",
        "_semantic_top3",
        "_semantic_prototype",
        "_semantic_score",
        "_bm25_score",
        "_rrf_score",
        "_rerank_score",
        "_final_score",
        "_best_match_phrase",
        "_positive_hits",
        "_negative_hits",
        "_constraint_evidence",
    ):
        if key in entry:
            result[key] = entry[key]

    return result


def build_user_message(question: str, relevant_entries: list[dict[str, Any]]) -> str:
    payload = {
        "task": (
            "Определи, какой из предложенных intent лучше всего соответствует "
            "вопросу клиента. Если выбранный intent действительно позволяет ответить "
            "по его базе знаний, сформируй краткий точный ответ."
        ),
        "decision_policy": [
            "Выбирать разрешено только один id из candidates.",
            "selected_kb_id обязателен при can_answer=true.",
            "Нельзя придумывать новый id.",
            "Оценивай намерение клиента, а не только совпадение отдельных слов.",
            "Сначала учитывай definition/scope/when_to_use/when_not_to_use/constraints/negative_phrases.",
            "positive_examples помогают понять формулировки, но сами по себе не доказывают intent.",
            "Если несколько кандидатов остаются действительно неразличимыми, используй needs_clarification=true.",
            "При needs_clarification=true обязательно укажи clarification_kb_id из candidates.",
            "Если ни один кандидат не подходит, can_answer=false и needs_clarification=false.",
            "Если вопрос является бессмысленным мусором, is_noise=true.",
            "Не выдумывай факты, которых нет в переданных кандидатах.",
            "Не используй сведения из других категорий и не смешивай ответы нескольких кандидатов.",
        ],
        "question": question,
        "candidates": [_entry_for_llm(entry) for entry in relevant_entries],
        "output_schema": {
            "is_noise": "boolean",
            "can_answer": "boolean",
            "selected_kb_id": "string|null",
            "needs_clarification": "boolean",
            "clarification_kb_id": "string|null",
            "reason": "string|null",
            "answer": "string|null",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_message_text(response: Any) -> str:
    """Extract the first message text from the Responses API object."""
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            # Some SDK/model combinations expose content as dict-like values.
            if isinstance(content, dict):
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()
    raise ValueError("LLM response does not contain a message text")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    if lines and lines[0].strip().lower() == "json":
        lines = lines[1:]
    return "\n".join(lines).strip()


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
        if not question.strip():
            return LLMResult(False, None, "empty_question")

        candidates = [
            entry for entry in relevant_entries
            if entry.get("id")
        ]
        if not candidates:
            return LLMResult(False, None, "no_candidates")

        candidate_ids = {str(entry["id"]) for entry in candidates}
        user_message = build_user_message(question, candidates)

        try:
            # Current compatibility path: rely on the stored Yandex prompt for the
            # model/system setup and ask for strict JSON in the user payload.
            response = await self._client.responses.create(
                prompt={"id": self.prompt_id},
                input=user_message,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "intent_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "is_noise": {"type": ["boolean"]},
                                "can_answer": {"type": ["boolean"]},
                                "selected_kb_id": {"type": ["string", "null"]},
                                "needs_clarification": {"type": ["boolean"]},
                                "clarification_kb_id": {"type": ["string", "null"]},
                                "reason": {"type": ["string", "null"]},
                                "answer": {"type": ["string", "null"]},
                            },
                            "required": [
                                "is_noise",
                                "can_answer",
                                "selected_kb_id",
                                "needs_clarification",
                                "clarification_kb_id",
                                "reason",
                                "answer",
                            ],
                        },
                    }
                },
            )

            text = _strip_json_fence(_extract_message_text(response))
            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError("LLM output must be a JSON object")

            is_noise = bool(data.get("is_noise", False))
            can_answer = bool(data.get("can_answer", False))
            needs_clarification = bool(data.get("needs_clarification", False))

            selected_kb_id = data.get("selected_kb_id")
            selected_kb_id = (
                str(selected_kb_id).strip()
                if selected_kb_id is not None and str(selected_kb_id).strip()
                else None
            )

            clarification_kb_id = data.get("clarification_kb_id")
            clarification_kb_id = (
                str(clarification_kb_id).strip()
                if clarification_kb_id is not None and str(clarification_kb_id).strip()
                else None
            )

            answer = data.get("answer")
            answer = answer.strip() if isinstance(answer, str) and answer.strip() else None

            reason = data.get("reason")
            reason = reason.strip() if isinstance(reason, str) and reason.strip() else None

            logger.info(
                "LLM response | noise=%s | can_answer=%s | selected=%s | "
                "clarification=%s | reason=%s",
                is_noise,
                can_answer,
                selected_kb_id,
                clarification_kb_id,
                reason,
            )

            if is_noise:
                return LLMResult(False, None, reason, is_noise=True)

            # Never accept a model-produced answer without a valid selected intent.
            if can_answer:
                if selected_kb_id not in candidate_ids:
                    logger.warning(
                        "LLM selected invalid candidate | selected=%s | candidates=%s",
                        selected_kb_id,
                        sorted(candidate_ids),
                    )
                    return LLMResult(
                        False,
                        None,
                        "invalid_selected_kb_id",
                    )

                if not answer:
                    return LLMResult(
                        False,
                        None,
                        "missing_answer",
                        selected_kb_id=selected_kb_id,
                    )

                # If the model says can_answer, clarification must be false.
                if needs_clarification:
                    needs_clarification = False

                return LLMResult(
                    can_answer=True,
                    answer=answer + OPERATOR_HINT,
                    reason=reason,
                    selected_kb_id=selected_kb_id,
                )

            if needs_clarification:
                if clarification_kb_id not in candidate_ids:
                    logger.warning(
                        "LLM selected invalid clarification candidate | selected=%s | candidates=%s",
                        clarification_kb_id,
                        sorted(candidate_ids),
                    )
                    return LLMResult(
                        False,
                        None,
                        "invalid_clarification_kb_id",
                    )

                return LLMResult(
                    can_answer=False,
                    answer=None,
                    reason=reason,
                    needs_clarification=True,
                    clarification_kb_id=clarification_kb_id,
                )

            return LLMResult(
                can_answer=False,
                answer=None,
                reason=reason or "no_suitable_candidate",
            )

        except Exception as exc:
            logger.exception("LLM call/parse failed: %s", exc)
            return LLMResult(
                can_answer=False,
                answer=None,
                reason="llm_unavailable_or_invalid_output",
            )
