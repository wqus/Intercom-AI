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
    payload["created_at"] = 201
    msg = parse(payload)
    assert msg is not None
    # For a later `conversation.user.replied`, Intercom may keep the original
    # conversation source while the actual triggering message appears in
    # conversation_parts. The current user part must win.
    assert msg.text == "Более новое сообщение"
    assert msg.updated_at == "200"
    assert msg.raw["_provider_part_id"] is None or msg.raw["_provider_part_id"] == ""


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
        "created_at": 201,
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


def test_webhook_event_order_falls_back_to_notification_time_when_source_has_no_message():
    from app.infrastructure.intercom.parser import event_order_at as _event_order_at
    assert _event_order_at({
        "created_at": 999,
        "topic": "conversation.user.replied",
        "data": {"item": {"source": {"created_at": 123}}},
    }) == 999


def test_webhook_event_order_falls_back_to_notification_time():
    from app.infrastructure.intercom.parser import event_order_at as _event_order_at
    assert _event_order_at({
        "created_at": 456,
        "topic": "conversation.user.replied",
        "data": {"item": {"source": {}}},
    }) == 456


def test_ticket_created_remains_parseable_for_dedicated_ticket_workflow():
    # The webhook layer gates ticket.created by PROCESS_TICKET_CREATED; parser itself
    # remains backward-compatible for explicit ticket workflows.
    payload = {
        "id": "ticket-1",
        "topic": "ticket.created",
        "data": {"item": {
            "id": "t-1",
            "updated_at": "123",
            "ticket_attributes": {"_default_description_": "Описание заявки"},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.entity_id == "t-1"
    assert msg.text == "Описание заявки"


def test_operator_reply_uses_current_user_part_when_source_is_stale_original_message():
    payload = {
        "id": "evt-operator",
        "topic": "conversation.user.replied",
        "created_at": 1788877044,
        "data": {"item": {
            "id": "conv-operator",
            "source": {
                "type": "conversation",
                "id": "conv-operator",
                "body": "123 123 123",
                "created_at": 1788876988,
                "author": {"type": "user", "name": "User"},
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "old", "body": "123 123 123", "created_at": 1788876988, "author": {"type": "user"}},
                {"id": "op-1", "body": "Оператор", "created_at": 1788877043, "author": {"type": "user"}},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == "Оператор"
    assert msg.raw["_provider_part_id"] == "op-1"
    assert msg.event_key == "message:conv-operator:op-1"


def test_later_conversation_reply_does_not_reuse_original_source_id_as_dedup_key():
    payload = {
        "id": "notif-2",
        "topic": "conversation.user.replied",
        "created_at": 201,
        "data": {"item": {
            "id": "conv-1",
            "source": {
                "type": "conversation",
                "id": "3943732434",
                "body": "Когда запуск потока?",
                "created_at": 100,
                "author": {"type": "user"},
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "part-2", "body": "Аа, понял. А с онлайн школами чо", "created_at": 200, "author": {"type": "user"}},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == "Аа, понял. А с онлайн школами чо"
    assert msg.raw["_provider_part_id"] == "part-2"
    assert msg.event_key == "message:conv-1:part-2"
    from app.infrastructure.intercom.parser import event_order_at
    assert event_order_at(payload) == 200


def test_conversation_and_ticket_events_share_part_idempotency_key():
    conv_payload = {
        "id": "conv-event",
        "topic": "conversation.user.replied",
        "created_at": 1788876990,
        "data": {"item": {
            "id": "same-entity",
            "source": {"type": "admin_issued", "author": {"type": "admin"}},
            "conversation_parts": {"conversation_parts": [
                {"id": "shared-part", "body": "123 12", "created_at": 1788876988, "author": {"type": "user"}},
            ]},
        }},
    }
    ticket_payload = {
        "id": "ticket-event",
        "topic": "ticket.contact.replied",
        "created_at": 1788876990,
        "data": {"item": {
            "id": "same-entity",
            "ticket": {"id": "same-entity"},
            "ticket_part": {"id": "shared-part", "body": "123 12", "created_at": 1788876988, "author": {"type": "user"}},
        }},
    }
    conv = parse(conv_payload)
    ticket = parse(ticket_payload)
    assert conv is not None and ticket is not None
    assert conv.event_key == ticket.event_key


def test_event_order_uses_actual_user_part_when_conversation_source_is_admin_initiated():
    from app.infrastructure.intercom.parser import event_order_at
    payload = {
        "id": "evt-real",
        "topic": "conversation.user.replied",
        "created_at": 1788877044,
        "data": {"item": {
            "source": {
                "type": "admin_initiated",
                "created_at": 1788775828,
                "author": {"type": "admin", "id": "7848715"},
                "body": "",
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "old", "created_at": 1788876988, "body": "old", "author": {"type": "user"}},
                {"id": "current", "created_at": 1788877043, "body": "Оператор", "author": {"type": "user"}},
            ]},
        }},
    }
    assert event_order_at(payload) == 1788877043


def test_realistic_admin_initiated_source_uses_current_user_part_for_ordering_and_text():
    from app.infrastructure.intercom.parser import event_order_at
    payload = {
        "id": "notif-realistic",
        "created_at": 1788877044,
        "topic": "conversation.user.replied",
        "data": {"item": {
            "id": "215475824210454",
            "source": {
                "type": "admin_initiated",
                "id": "215475824210454",
                "delivered_as": "admin_initiated",
                "body": "",
                "author": {"type": "admin", "id": "7848715"},
            },
            "conversation_parts": {"conversation_parts": [
                {"id": "old", "part_type": "comment", "body": "старое", "created_at": 1788876988, "author": {"type": "user", "id": "u1"}},
                {"id": "current", "part_type": "comment", "body": "Оператор", "created_at": 1788877043, "author": {"type": "user", "id": "u1"}},
            ]},
        }},
    }
    msg = parse(payload)
    assert msg is not None
    assert msg.text == "Оператор"
    assert msg.raw["_provider_part_id"] == "current"
    assert event_order_at(payload) == 1788877043
