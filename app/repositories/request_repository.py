from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import (
    EscalationReason,
    Request,
    RequestResult,
    RequestType,
)


class RequestRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        conversation_id: str,
        request_type: RequestType,
        result: RequestResult,
        question: str,
        answer: str | None = None,
        category: str | None = None,
        escalation_reason: EscalationReason | None = None,
        predicted_kb_id: str | None = None,
        subquestion: str | None = None,
        score: float | None = None,
        second_score: float | None = None,
        margin: float | None = None,
        embedding_score: float | None = None,
        intent_fit_score: float | None = None,
        decision_reason: str | None = None,
    ) -> Request:
        request = Request(
            conversation_id=conversation_id,
            type=request_type,
            result=result,
            escalation_reason=escalation_reason,
            predicted_kb_id=predicted_kb_id,
            category=category,
            question=question,
            subquestion=subquestion,
            answer=answer,
            score=score,
            second_score=second_score,
            margin=margin,
            embedding_score=embedding_score,
            intent_fit_score=intent_fit_score,
            decision_reason=decision_reason,
        )

        self.session.add(request)
        await self.session.commit()
        await self.session.refresh(request)

        return request