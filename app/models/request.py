from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class RequestType(str, Enum):
    CONVERSATION = "conversation"
    TICKET = "ticket"


class RequestResult(str, Enum):
    TEMPLATE = "template"
    LLM = "llm"
    ESCALATED = "escalated"
    IGNORED = "ignored"


class EscalationReason(str, Enum):
    USER_REQUESTED_OPERATOR = "user_requested_operator"
    NO_CATEGORY = "no_category"
    LLM_CANNOT_ANSWER = "llm_cannot_answer"
    INTERNAL_ERROR = "internal_error"
    WANTS_NEW_SCHOOL = "wants_new_school"
    ATTACHMENTS = "attachments"

class Request(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)

    conversation_id: Mapped[str] = mapped_column(String(64), index=True)

    type: Mapped[RequestType] = mapped_column(
        SqlEnum(RequestType),
    )

    result: Mapped[RequestResult] = mapped_column(
        SqlEnum(RequestResult),
    )

    escalation_reason: Mapped[EscalationReason | None] = mapped_column(
        SqlEnum(EscalationReason),
        nullable=True,
    )

    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    question: Mapped[str] = mapped_column(
        Text,
    )

    answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )