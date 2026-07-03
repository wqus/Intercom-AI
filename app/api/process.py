import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from app.services.knowledge_base import KnowledgeBase, MatchMode
from app.services.llm_client import YandexLLMClient
from app.core.dependencies import get_llm_client, get_knowledge_base

logger = logging.getLogger("process")

router = APIRouter()


class ProcessRequest(BaseModel):
    question: str

    @field_validator("question")
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > 2000:
            raise ValueError("question too long (max 2000 chars)")
        return v


class ProcessResponse(BaseModel):
    question: str
    decision: str
    answer: str | None
    matched_category: str | None
    kb_score: float
    escalation_reason: str | None


@router.post("/process", response_model=ProcessResponse)
async def process_question(
    body: ProcessRequest,
    llm: YandexLLMClient = Depends(get_llm_client),
    kb: KnowledgeBase = Depends(get_knowledge_base),
):
    question = body.question
    logger.info("Processing question: %s", question[:80])

    match = kb.match(question)

    if match.mode == MatchMode.TEMPLATE:
        return ProcessResponse(
            question=question,
            decision="template",
            answer=match.template_answer,
            matched_category=match.category,
            kb_score=match.score,
            escalation_reason=None,
        )

    if match.mode == MatchMode.NO_MATCH or not match.relevant_entries:
        return ProcessResponse(
            question=question,
            decision="escalate",
            answer=None,
            matched_category=None,
            kb_score=match.score,
            escalation_reason="no_kb_match",
        )

    llm_result = await llm.classify(
        question=question,
        relevant_entries=match.relevant_entries,
    )

    if llm_result.can_answer and llm_result.answer:
        return ProcessResponse(
            question=question,
            decision="llm_answer",
            answer=llm_result.answer,
            matched_category=match.category,
            kb_score=match.score,
            escalation_reason=None,
        )

    return ProcessResponse(
        question=question,
        decision="escalate",
        answer=None,
        matched_category=match.category,
        kb_score=match.score,
        escalation_reason=llm_result.reason,
    )