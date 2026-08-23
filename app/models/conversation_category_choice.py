from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class CategoryChoiceStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ConversationCategoryChoice(Base):
    __tablename__ = "conversation_category_choices"

    id: Mapped[int] = mapped_column(primary_key=True)

    conversation_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
    )

    question: Mapped[str] = mapped_column(Text)

    candidate_1: Mapped[str] = mapped_column(String(64))
    candidate_2: Mapped[str] = mapped_column(String(64))
    candidate_3: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    status: Mapped[CategoryChoiceStatus] = mapped_column(
        SqlEnum(CategoryChoiceStatus),
        default=CategoryChoiceStatus.PENDING,
        nullable=False,
        index=True,
    )

    attempts: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    selected_candidates: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
