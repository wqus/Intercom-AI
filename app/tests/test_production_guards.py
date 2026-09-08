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


def test_outbound_action_table_is_registered_and_unique():
    from app.database.base import Base
    table = Base.metadata.tables["outbound_actions"]
    assert "action_key" in table.c
    assert table.c.action_key.unique is True
    assert "status" in table.c




def test_cpu_only_torch_dependency_is_pinned_for_production():
    from pathlib import Path

    requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.13.0+cpu" in requirements
    assert "torch==2.13.0\n" not in requirements


def test_event_repository_count_for_entity_uses_sqlalchemy_count_expression():
    from app.infrastructure.persistence.repository import EventRepository
    from sqlalchemy.sql import Select

    class Session:
        async def scalar(self, stmt):
            assert isinstance(stmt, Select)
            return 3

    import asyncio
    assert asyncio.run(EventRepository(Session()).count_for_entity("conv")) == 3


def test_ticket_created_is_processed_as_initial_customer_request():
    from app.infrastructure.intercom.parser import parse
    payload = {
        "id": "notif-ticket-created",
        "topic": "ticket.created",
        "created_at": 1788895801,
        "data": {"item": {
            "id": "215475848641277",
            "updated_at": 1788895800,
            "ticket_attributes": {
                "_default_title_": "Проверка",
                "_default_description_": "Перейти в другую шк хочу",
            },
        }},
    }
    message = parse(payload)
    assert message is not None
    assert message.channel.value == "ticket"
    assert message.text == "Перейти в другую шк хочу"


def test_same_ticket_comment_and_conversation_part_have_one_global_event_key():
    from app.infrastructure.intercom.parser import parse
    conv = parse({
        "id": "conv-event",
        "topic": "conversation.user.replied",
        "created_at": 200,
        "data": {"item": {
            "id": "conversation-1",
            "source": {"type": "admin_initiated", "author": {"type": "admin"}},
            "conversation_parts": {"conversation_parts": [
                {"id": "part-global-1", "body": "хочу в другую школу", "created_at": 199,
                 "author": {"type": "user"}},
            ]},
        }},
    })
    ticket = parse({
        "id": "ticket-event",
        "topic": "ticket.contact.replied",
        "created_at": 200,
        "data": {"item": {
            "id": "ticket-1",
            "ticket": {"id": "ticket-1"},
            "ticket_part": {"id": "part-global-1", "body": "хочу в другую школу", "created_at": 199,
                            "author": {"type": "user"}},
        }},
    })
    assert conv is not None and ticket is not None
    assert conv.event_key == ticket.event_key == "message:part:part-global-1"
