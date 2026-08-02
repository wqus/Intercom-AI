import uuid
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database.base import Base


class PendingClarification(Base):
    __tablename__ = "pending_clarifications"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    category_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)