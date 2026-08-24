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

