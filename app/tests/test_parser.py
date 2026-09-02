from app.infrastructure.intercom.parser import parse

def test_conversation_event_is_normalized():
    payload = {
        "id": "evt-1",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-1",
            "updated_at": "123",
            "source": {"body": "<p>Не могу записаться</p>", "author": {"name":"Ben","email":"x@y.z"}},
            "conversation_parts": {"conversation_parts":[]},
        }},
    }
    msg=parse(payload)
    assert msg is not None
    assert msg.entity_id=="conv-1"
    assert msg.text=="Не могу записаться"
    assert msg.event_key.startswith("conversation.user.replied:conv-1:")

def test_unknown_topic_is_ignored():
    assert parse({"id":"1","topic":"conversation.admin.closed","data":{"item":{"id":"1"}}}) is None



def test_conversation_user_replied_does_not_fallback_to_admin_source():
    payload = {
        "id": "evt-2",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-2",
            "updated_at": "123",
            "source": {"body": "admin reply", "author": {"type": "admin", "name": "Operator"}},
            "conversation_parts": {"conversation_parts": [
                {"body": "admin part", "author": {"type": "admin"}},
            ]},
        }},
    }
    assert parse(payload) is None


def test_conversation_parser_uses_latest_user_message():
    payload = {
        "id": "evt-old",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-replay",
            "source": {"body": "Первое сообщение", "author": {"type": "user", "name": "Ben"}, "created_at": "100"},
            "conversation_parts": {"conversation_parts": [
                {"body": "Первое сообщение", "author": {"type": "user"}, "created_at": "100"},
                {"body": "Более новое сообщение", "author": {"type": "user"}, "created_at": "200"},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    # For `conversation.user.replied`, the source is the event's message. Do
    # not replace it with a later historical user part.
    assert msg.text == "Первое сообщение"
    assert msg.updated_at == "100"


def test_attachments_belong_only_to_current_source_message():
    payload = {
        "id": "evt-new",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-attachments",
            "source": {"body": "Спасибо", "author": {"type": "user"}, "created_at": "200", "attachments": []},
            "conversation_parts": {"conversation_parts": [
                {"body": "Старое", "author": {"type": "user"}, "created_at": "100", "attachments": [{"id": "a1"}]},
                {"body": "Спасибо", "author": {"type": "user"}, "created_at": "200", "attachments": []},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == "Спасибо"
    assert msg.has_attachments is False


def test_conversation_user_replied_prefers_event_source_over_later_history():
    payload = {
        "id": "evt-current",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-order",
            "source": {
                "body": "1",
                "author": {"type": "user", "name": "User"},
                "created_at": "200",
            },
            "conversation_parts": {"conversation_parts": [
                {"body": "Первый вопрос", "author": {"type": "user"}, "created_at": "100"},
                # Simulates a later message already visible in conversation history
                # when a delayed webhook is delivered.
                {"body": "Оператор", "author": {"type": "user"}, "created_at": "300"},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == "1"
    assert msg.updated_at == "200"
