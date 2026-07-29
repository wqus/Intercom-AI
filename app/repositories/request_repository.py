from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request import (
    Request,
    RequestResult,
    RequestType,
    EscalationReason,
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
    ) -> Request:
        request = Request(
            conversation_id=conversation_id,
            type=request_type,
            result=result,
            escalation_reason=escalation_reason,
            category=category,
            question=question,
            answer=answer,
        )

        self.session.add(request)
        await self.session.commit()