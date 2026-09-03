import asyncio

from app.application.support_service import SupportApplication
from app.domain.enums import Channel, ConversationStateKind, Decision, EscalationReason
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction


class FakeState:
    def __init__(self, state=ConversationStateKind.IDLE, data=None):
        self.state = state
        self.data = data if data is not None else {}


class FakeStates:
    instances = None

    def __init__(self, session):
        self.state = FakeState(FakeStates.instances.state, FakeStates.instances.data)
        self.replacements = FakeStates.instances.replacements

    async def get(self, entity_id, *, for_update=False):
        return self.state

    async def replace(self, entity_id, state, data):
        self.state.state = state
        self.state.data = dict(data)
        self.replacements.append((state, dict(data)))
        FakeStates.instances.state = state
        FakeStates.instances.data = dict(data)


class FakeEscalations:
    def __init__(self, session):
        self.rows = []

    async def is_escalated(self, entity_id):
        return False

    async def record(self, **kwargs):
        self.rows.append(kwargs)
        return kwargs


class FakeIntercom:
    def __init__(self):
        self.replies = []

    async def send_reply(self, entity_id, channel, text):
        self.replies.append(text)
        return True


class FakeOutbox:
    def __init__(self, *args, **kwargs):
        self.actions = []

    async def enqueue(self, **kwargs):
        self.actions.append(kwargs)
        return True


class FakePipeline:
    def __init__(self, responses):
        self.responses = list(responses)

    async def decide(self, message):
        return self.responses.pop(0)



def result(decision, text="Пожалуйста, уточните вопрос подробнее."):
    return ProcessingResult(
        traces=[DecisionTrace(decision=decision, question="test")],
        actions=[OutboundAction("reply", text)],
    )


def install_repos(monkeypatch, initial_state):
    import app.application.support_service as module
    FakeStates.instances = type("Holder", (), {
        "state": initial_state[0],
        "data": initial_state[1] if initial_state[1] is None else dict(initial_state[1]),
        "replacements": [],
    })()
    monkeypatch.setattr(module, "StateRepository", FakeStates)
    monkeypatch.setattr(module, "EscalatedConversationRepository", FakeEscalations)
    monkeypatch.setattr(module, "InteractionRepository", lambda session: type(
        "Repo", (), {"add_many": lambda self, *args, **kwargs: asyncio.sleep(0)}
    )())


def test_rephrase_asks_twice_then_escalates(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.REPHRASING, {"attempts": 1}))
    app = SupportApplication(
        FakePipeline([result(Decision.REPHRASE), result(Decision.REPHRASE)]),
        FakeIntercom(),
        None,
        FakeOutbox(),
    )
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "непонятно")

    class Event: id = 1

    # First failed clarification -> second request; state must become 2.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.REPHRASING
    assert FakeStates.instances.data["attempts"] == 2

    # Third failed message -> escalation path, no third rephrase from pipeline.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.ESCALATED
    assert FakeStates.instances.data["reason"] == "clarification_exhausted"


def test_successful_answer_resets_state(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.REPHRASING, {"attempts": 1}))
    app = SupportApplication(
        FakePipeline([
            result(Decision.ANSWER_TEMPLATE, "Готово"),
        ]),
        FakeIntercom(),
        None,
        FakeOutbox(),
    )
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "нормальный вопрос")

    class Event: id = 1

    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE
    assert FakeStates.instances.data == {}


def test_initial_rephrase_then_two_requests_then_escalation(monkeypatch):
    # Initial user message was already answered with a clarification request.
    install_repos(monkeypatch, (ConversationStateKind.IDLE, {}))
    app = SupportApplication(
        FakePipeline([
            result(Decision.REPHRASE),
            result(Decision.REPHRASE),
        ]),
        FakeIntercom(),
        None,
        FakeOutbox(),
    )
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "что-то странное")

    class Event: id = 1

    # Initial ambiguous request -> clarification #1.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.REPHRASING
    assert FakeStates.instances.data["attempts"] == 1

    # First failed follow-up -> clarification #2.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.REPHRASING
    assert FakeStates.instances.data["attempts"] == 2

    # Second failed follow-up = third user message -> escalation.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.ESCALATED
    assert FakeStates.instances.data["reason"] == "clarification_exhausted"


def test_escalated_conversation_never_reaches_pipeline(monkeypatch):
    import app.application.support_service as module

    class BlockingEscalations:
        def __init__(self, session):
            pass
        async def is_escalated(self, entity_id):
            return True

    class NeverCalledPipeline:
        async def decide(self, message):
            raise AssertionError("pipeline must not run for an escalated conversation")

    monkeypatch.setattr(module, "EscalatedConversationRepository", BlockingEscalations)
    app = SupportApplication(NeverCalledPipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "новое сообщение")

    class Event: id = 1

    asyncio.run(app.handle(object(), Event(), message))


def test_short_one_word_clarification_reply_is_resolved_contextually(monkeypatch):
    import app.application.support_service as module
    from app.domain.enums import ConversationStateKind

    install_repos(monkeypatch, (ConversationStateKind.CLARIFYING, {"kb_id": "kb_005", "attempts": 1}))

    class KB:
        def get_short_clarification_routes(self, kb_id):
            assert kb_id == "kb_005"
            return {"уйти": "kb_005", "перейти": "kb_006", "новую": "kb_006"}
        def get_by_id(self, kb_id):
            return {"id": kb_id, "template_answer": f"ANSWER:{kb_id}", "choice_label": kb_id}

    class Pipeline:
        kb = KB()
        async def decide(self, message):
            raise AssertionError("single-word contextual reply must not enter generic classifier")

    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "уйти")

    class Event: id = 99

    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE
    assert FakeStates.instances.data == {}


def test_choice_state_answers_from_persisted_snapshot(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.CHOOSING, {
        "candidate_ids": ["kb_005", "kb_006"],
        "candidates": [
            {"kb_id": "kb_005", "label": "Выйти", "answer": "SNAPSHOT-EXIT"},
            {"kb_id": "kb_006", "label": "Перейти", "answer": "SNAPSHOT-TRANSFER"},
        ],
        "attempts": 0,
    }))

    class KB:
        @staticmethod
        def parse_category_choice(text, max_index):
            return [1] if text == "1" and max_index == 2 else []
        def get_by_id(self, kb_id):
            # Simulate a KB mutation after the menu was shown. The saved snapshot
            # must still resolve the user's numbered choice deterministically.
            return {"id": kb_id, "template_answer": "MUTATED"}

    class Pipeline:
        kb = KB()
        async def decide(self, message):
            raise AssertionError("choice reply must be consumed by CHOOSING state")

    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "1")

    class Event: id = 100

    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE


def test_unknown_conversation_state_is_recovered_to_idle(monkeypatch):
    install_repos(monkeypatch, ("future_state_not_implemented", {}))

    called = {"count": 0}
    class Pipeline:
        async def decide(self, message):
            called["count"] += 1
            from app.domain.models import ProcessingResult
            return ProcessingResult()

    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "нормальный вопрос")

    class Event: id = 101

    # Unknown/corrupt state must not silently consume the event: it is reset to
    # the root state and the current turn is re-evaluated normally.
    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE


def test_choice_invalid_free_text_does_not_auto_escalate(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.CHOOSING, {
        "candidate_ids": ["kb_005", "kb_006"],
        "candidates": [
            {"kb_id": "kb_005", "label": "Выйти", "answer": "A"},
            {"kb_id": "kb_006", "label": "Перейти", "answer": "B"},
        ],
        "attempts": 2,
    }))
    class KB:
        def parse_category_choice(self, text, max_index):
            return []
    class Pipeline:
        kb = KB()
        async def decide(self, message):
            return ProcessingResult(
                traces=[DecisionTrace(decision=Decision.IGNORE, question=message.text)],
                actions=[],
            )
    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "это вообще про другое")
    class Event: id = 100

    class Session:
        async def flush(self):
            return None
    asyncio.run(app.handle(Session(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE
    assert FakeStates.instances.data == {}



def test_top3_round_trip_always_resolves_numeric_choice(monkeypatch):
    """The exact user-visible Top-3 must survive one full two-event round trip."""
    install_repos(monkeypatch, (ConversationStateKind.IDLE, {}))

    class KB:
        def get_by_id(self, kb_id):
            return {
                "id": kb_id,
                "choice_label": {"kb_a": "Вариант A", "kb_b": "Вариант B"}[kb_id],
                "template_answer": {"kb_a": "ANSWER A", "kb_b": "ANSWER B"}[kb_id],
            }

        def parse_category_choice(self, text, max_index):
            return [int(text)] if text in {"1", "2"} and int(text) <= max_index else []

    first = ProcessingResult(
        traces=[DecisionTrace(
            decision=Decision.CHOICE,
            question="исходный вопрос",
            choice_candidate_ids=["kb_a", "kb_b"],
            choice_candidates=[
                {"kb_id": "kb_a", "label": "Вариант A", "answer": "ANSWER A"},
                {"kb_id": "kb_b", "label": "Вариант B", "answer": "ANSWER B"},
            ],
        )],
        actions=[OutboundAction("reply", "MENU")],
    )

    class Pipeline:
        kb = KB()
        def __init__(self): self.calls = 0
        async def decide(self, message):
            self.calls += 1
            if self.calls == 1:
                return first
            raise AssertionError("numeric choice must be consumed by CHOOSING state")

    class Event: id = 777

    outbox = FakeOutbox()
    pipeline = Pipeline()
    app = SupportApplication(pipeline, FakeIntercom(), None, outbox)
    initial = InboundMessage("e1", "k1", "topic", "c", Channel.CONVERSATION, "исходный вопрос")
    asyncio.run(app.handle(object(), Event(), initial))

    assert FakeStates.instances.state == ConversationStateKind.CHOOSING
    assert FakeStates.instances.data["candidate_ids"] == ["kb_a", "kb_b"]
    assert [x["answer"] for x in FakeStates.instances.data["candidates"]] == ["ANSWER A", "ANSWER B"]
    assert any(a["payload"]["text"] == "MENU" for a in outbox.actions)

    choice = InboundMessage("e2", "k2", "topic", "c", Channel.CONVERSATION, "2")
    asyncio.run(app.handle(object(), Event(), choice))

    assert FakeStates.instances.state == ConversationStateKind.IDLE
    assert any("ANSWER B" in a["payload"]["text"] for a in outbox.actions)


def test_top3_is_never_sent_when_choice_state_cannot_be_persisted(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.IDLE, {}))

    class FailingStates(FakeStates):
        async def replace(self, entity_id, state, data):
            raise RuntimeError("db write failed")

    import app.application.support_service as module
    monkeypatch.setattr(module, "StateRepository", FailingStates)
    monkeypatch.setattr(module, "EscalatedConversationRepository", FakeEscalations)

    class KB:
        def get_by_id(self, kb_id):
            return {"id": kb_id, "choice_label": kb_id, "template_answer": "ANSWER"}

    class Pipeline:
        kb = KB()
        async def decide(self, message):
            return ProcessingResult(
                traces=[DecisionTrace(
                    decision=Decision.CHOICE, question=message.text,
                    choice_candidate_ids=["a", "b"],
                    choice_candidates=[
                        {"kb_id": "a", "label": "A", "answer": "ANSWER A"},
                        {"kb_id": "b", "label": "B", "answer": "ANSWER B"},
                    ],
                )],
                actions=[OutboundAction("reply", "MENU")],
            )

    outbox = FakeOutbox()
    app = SupportApplication(Pipeline(), FakeIntercom(), None, outbox)
    message = InboundMessage("e", "k", "topic", "c", Channel.CONVERSATION, "вопрос")
    class Event: id = 778

    try:
        asyncio.run(app.handle(object(), Event(), message))
        assert False, "state persistence failure must propagate"
    except RuntimeError as exc:
        assert "db write failed" in str(exc)
    assert not any(a["payload"]["text"] == "MENU" for a in outbox.actions)


def test_top3_plus_same_turn_escalation_never_sends_unresolvable_menu(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.IDLE, {}))

    class Pipeline:
        class KB:
            def get_by_id(self, kb_id):
                return {"id": kb_id, "choice_label": kb_id, "template_answer": "ANSWER"}
        kb = KB()
        async def decide(self, message):
            return ProcessingResult(
                traces=[
                    DecisionTrace(
                        decision=Decision.CHOICE, question="q1",
                        choice_candidate_ids=["a", "b"],
                        choice_candidates=[
                            {"kb_id": "a", "label": "A", "answer": "A"},
                            {"kb_id": "b", "label": "B", "answer": "B"},
                        ],
                    ),
                    DecisionTrace(
                        decision=Decision.ESCALATE, question="q2",
                        escalation_reason=EscalationReason.NO_CATEGORY,
                    ),
                ],
                actions=[
                    OutboundAction("reply", "MENU"),
                    OutboundAction("reply", "ESC", True),
                ],
            )

    outbox = FakeOutbox()
    app = SupportApplication(Pipeline(), FakeIntercom(), None, outbox)
    msg = InboundMessage("e", "k", "topic", "c", Channel.CONVERSATION, "два вопроса")
    class Event: id = 779
    asyncio.run(app.handle(object(), Event(), msg))

    texts = [a["payload"]["text"] for a in outbox.actions if a["action_type"] == "intercom_reply"]
    assert texts
    assert all("MENU" not in text for text in texts)
    assert FakeStates.instances.state == ConversationStateKind.ESCALATED


def test_operator_is_checked_before_state_and_classifier(monkeypatch):
    """Explicit operator request has absolute priority over persisted state."""
    install_repos(monkeypatch, (ConversationStateKind.CHOOSING, {
        "candidate_ids": ["kb_a", "kb_b"],
        "candidates": [
            {"kb_id": "kb_a", "label": "A", "answer": "ANSWER A"},
            {"kb_id": "kb_b", "label": "B", "answer": "ANSWER B"},
        ],
        "attempts": 0,
    }))

    class Pipeline:
        class KB:
            def parse_category_choice(self, text, max_index):
                raise AssertionError("state parser must not run before operator check")
        kb = KB()
        async def decide(self, message):
            raise AssertionError("classifier must not run for explicit operator request")

    called = {}
    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    async def fake_escalate(session, message, event, reason, reply_text=None):
        called["reason"] = reason
    app._escalate = fake_escalate

    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "оператор")
    class Event: id = 801

    asyncio.run(app.handle(object(), Event(), message))
    assert called["reason"] == EscalationReason.USER_REQUESTED_OPERATOR


def test_numeric_choice_is_consumed_by_state_without_classifier(monkeypatch):
    """A persisted Top-3 must make `1/2/3` deterministic and never reclassify it."""
    install_repos(monkeypatch, (ConversationStateKind.CHOOSING, {
        "candidate_ids": ["kb_a", "kb_b", "kb_c"],
        "candidates": [
            {"kb_id": "kb_a", "label": "A", "answer": "ANSWER A"},
            {"kb_id": "kb_b", "label": "B", "answer": "ANSWER B"},
            {"kb_id": "kb_c", "label": "C", "answer": "ANSWER C"},
        ],
        "attempts": 0,
    }))

    class Pipeline:
        class KB:
            def parse_category_choice(self, text, max_index):
                return [int(text)] if text.isdigit() and 1 <= int(text) <= max_index else []
            def get_by_id(self, kb_id):
                return {"id": kb_id, "template_answer": f"ANSWER {kb_id[-1].upper()}"}
        kb = KB()
        async def decide(self, message):
            raise AssertionError("numeric choice must never reach generic classifier")

    outbox = FakeOutbox()
    app = SupportApplication(Pipeline(), FakeIntercom(), None, outbox)
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "2")
    class Event: id = 802

    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE
    texts = [
        x["payload"]["text"]
        for x in outbox.actions
        if x["action_type"] == "intercom_reply"
    ]
    assert texts and "ANSWER B" in texts[-1]
    assert "ANSWER A" not in texts[-1]


def test_top3_persists_state_before_outbox_and_keeps_snapshot(monkeypatch):
    """The menu is created only after state persistence, never the other way around."""
    install_repos(monkeypatch, (ConversationStateKind.IDLE, {}))

    events = []

    class KB:
        def get_by_id(self, kb_id):
            return {
                "id": kb_id,
                "choice_label": kb_id,
                "template_answer": f"ANSWER:{kb_id}",
            }

    class Pipeline:
        kb = KB()
        async def decide(self, message):
            return ProcessingResult(
                traces=[DecisionTrace(
                    decision=Decision.CHOICE,
                    question=message.text,
                    choice_candidate_ids=["kb_a", "kb_b"],
                    choice_candidates=[
                        {"kb_id": "kb_a", "label": "A", "answer": "A"},
                        {"kb_id": "kb_b", "label": "B", "answer": "B"},
                    ],
                )],
                actions=[OutboundAction("reply", "MENU")],
            )

    import app.application.support_service as module

    class OrderedStates(FakeStates):
        async def replace(self, entity_id, state, data):
            events.append(("state", state, dict(data)))
            await super().replace(entity_id, state, data)

    class OrderedOutbox(FakeOutbox):
        async def enqueue(self, **kwargs):
            assert FakeStates.instances.state == ConversationStateKind.CHOOSING
            assert FakeStates.instances.data["candidate_ids"] == ["kb_a", "kb_b"]
            events.append(("outbox", kwargs["action_type"]))
            return await super().enqueue(**kwargs)

    monkeypatch.setattr(module, "StateRepository", OrderedStates)
    outbox = OrderedOutbox()
    app = SupportApplication(Pipeline(), FakeIntercom(), None, outbox)
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "исходный вопрос")
    class Event: id = 803

    asyncio.run(app.handle(object(), Event(), message))
    assert events[0][0] == "state"
    assert any(kind == "outbox" for kind, *_ in events)
    assert events.index(next(item for item in events if item[0] == "state")) < events.index(next(item for item in events if item[0] == "outbox"))


def test_malformed_choosing_state_is_recovered_to_idle(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.CHOOSING, {"candidate_ids": ["only_one"]}))

    class Pipeline:
        async def decide(self, message):
            from app.domain.models import ProcessingResult
            return ProcessingResult()

    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "нормальный вопрос")

    class Event: id = 102

    asyncio.run(app.handle(object(), Event(), message))
    assert FakeStates.instances.state == ConversationStateKind.IDLE


def test_malformed_clarification_data_does_not_crash(monkeypatch):
    install_repos(monkeypatch, (ConversationStateKind.REPHRASING, None))

    called = {"count": 0}
    class Pipeline:
        async def decide(self, message):
            called["count"] += 1
            return ProcessingResult()

    app = SupportApplication(Pipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "нормальный вопрос")

    class Event: id = 103

    asyncio.run(app.handle(object(), Event(), message))
    assert called["count"] == 1
