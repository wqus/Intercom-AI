import asyncio

from app.application.support_service import SupportApplication
from app.domain.enums import Channel, ConversationStateKind, Decision
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction


class FakeState:
    def __init__(self, state=ConversationStateKind.IDLE, data=None):
        self.state = state
        self.data = data or {}


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
        "data": dict(initial_state[1]),
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
