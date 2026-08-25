from app.database.base import Base
from app.infrastructure.persistence.models import EscalatedConversation
from app.domain.enums import EscalationReason


def test_escalation_reason_is_persisted_as_string_not_postgres_enum():
    column = EscalatedConversation.__table__.c.reason
    assert column.type.length == 64
    assert column.nullable is False
    assert column.primary_key is False


def test_clarification_exhausted_is_part_of_domain_contract():
    assert EscalationReason.CLARIFICATION_EXHAUSTED.value == "clarification_exhausted"
    assert EscalationReason.CLARIFICATION_EXHAUSTED.value in {
        item.value for item in EscalationReason
    }


def test_escalation_blocklist_has_conversation_primary_key_and_indexes():
    table = EscalatedConversation.__table__
    assert table.primary_key.columns.keys() == ["conversation_id"]
    indexes = {index.name for index in table.indexes}
    assert "ix_escalated_conversations_reason" in indexes
    assert "ix_escalated_conversations_source_event_id" in indexes
    assert "ix_escalated_conversations_channel_reason" in indexes


def test_escalation_table_is_registered_in_metadata():
    assert "escalated_conversations" in Base.metadata.tables
