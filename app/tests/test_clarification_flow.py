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


def test_unknown_conversation_state_is_ignored(monkeypatch):
    install_repos(monkeypatch, ("future_state_not_implemented", {}))

    class NeverCallPipeline:
        async def decide(self, message):
            raise AssertionError("unknown state must not reach classifier")

    app = SupportApplication(NeverCallPipeline(), FakeIntercom(), None, FakeOutbox())
    message = InboundMessage("e", "k", "t", "c", Channel.CONVERSATION, "нормальный вопрос")

    class Event: id = 101

    # StateRepository in production is enum-backed, but the guard is validated
    # with a fake legacy/future state value here.
    asyncio.run(app.handle(object(), Event(), message))


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

