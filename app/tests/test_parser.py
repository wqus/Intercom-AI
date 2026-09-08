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


def test_attachment_only_user_message_is_not_dropped():
    payload = {
        "id": "evt-attachment-only",
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "conv-attachment-only",
            "source": {
                "body": "",
                "author": {"type": "user"},
                "created_at": "300",
                "attachments": [{"id": "a1"}],
            },
            "conversation_parts": {"conversation_parts": []},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == ""
    assert msg.has_attachments is True


def test_conversation_context_excludes_current_and_future_parts():
    from app.infrastructure.intercom.parser import _conversation_context
    current = {"id":"current", "body":"код", "created_at":30, "author":{"type":"user"}}
    item = {"conversation_parts": {"conversation_parts": [
        {"id":"u1","body":"Не могу зарегистрироваться","created_at":10,"author":{"type":"user"}},
        {"id":"a1","body":"На каком этапе?","created_at":20,"author":{"type":"admin"}},
        current,
        {"id":"u2","body":"Сколько стоит?","created_at":40,"author":{"type":"user"}},
    ]}}
    result = _conversation_context(item, "код", current)
    assert result == [
        {"role":"user","text":"Не могу зарегистрироваться"},
        {"role":"assistant","text":"На каком этапе?"},
    ]


def test_conversation_user_replied_uses_source_when_author_type_missing():
    payload = {
        "id": "evt-current",
        "topic": "conversation.user.replied",
        "data": {
            "item": {
                "id": "conv-1",
                "source": {"body": "Сколько стоит обучение?", "created_at": 200},
                "conversation_parts": {
                    "conversation_parts": [
                        {
                            "id": "old-user",
                            "body": "Не могу зарегистрироваться.",
                            "created_at": 100,
                            "author": {"type": "user", "name": "User"},
                        }
                    ]
                },
            }
        },
    }
    message = parse(payload)
    assert message is not None
    assert message.text == "Сколько стоит обучение?"


def test_webhook_event_order_uses_source_message_time():
    from app.infrastructure.intercom.parser import event_order_at as _event_order_at
    assert _event_order_at({
        "created_at": 999,
        "topic": "conversation.user.replied",
        "data": {"item": {"source": {"created_at": 123}}},
    }) == 123


def test_webhook_event_order_falls_back_to_notification_time():
    from app.infrastructure.intercom.parser import event_order_at as _event_order_at
    assert _event_order_at({
        "created_at": 456,
        "topic": "conversation.user.replied",
        "data": {"item": {"source": {}}},
    }) == 456
