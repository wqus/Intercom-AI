import asyncio

from app.application.support_service import SupportApplication
from app.domain.enums import Channel, Decision, EscalationReason
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction


def test_processing_result_escalation_state_is_derived_from_action():
    import asyncio
    from app.application.support_service import SupportApplication
    from app.domain.enums import Channel, Decision, EscalationReason
    from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction

    class FakePipeline:
        async def decide(self, message):
            return ProcessingResult(
                traces=[DecisionTrace(
                    decision=Decision.ESCALATE,
                    question=message.text,
                    escalation_reason=EscalationReason.USER_REQUESTED_OPERATOR,
                )],
                actions=[OutboundAction("reply", "Передаю обращение оператору.", True)],
                latency_ms=1.0,
            )

    class FakeState:
        def __init__(self):
            from app.domain.enums import ConversationStateKind
            self.state = ConversationStateKind.IDLE
            self.data = {}

    class FakeStates:
        def __init__(self, session):
            self.state = FakeState()
        async def get(self, entity_id, *, for_update=False):
            return self.state
        async def replace(self, entity_id, state, data):
            self.state.state = state
            self.state.data = data

    class FakeEscalations:
        async def is_escalated(self, entity_id):
            return False
        async def record(self, **kwargs):
            return kwargs

    class FakeInteraction:
        async def add_many(self, *args, **kwargs):
            return None

    class FakeOutbox:
        def __init__(self):
            self.actions = []
        async def enqueue(self, **kwargs):
            self.actions.append(kwargs)
            return True

    import app.application.support_service as module
    originals = (module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository)
    module.StateRepository = lambda session: FakeStates(session)
    module.InteractionRepository = lambda session: FakeInteraction()
    module.EscalatedConversationRepository = lambda session: FakeEscalations()
    try:
        outbox = FakeOutbox()
        app = SupportApplication(FakePipeline(), object(), None, outbox)
        message = InboundMessage("evt", "key", "conversation.user.replied", "conv", Channel.CONVERSATION, "оператор")
        event = type("Event", (), {"id": 1})()
        asyncio.run(app.handle(object(), event, message))
        assert app.outbox is outbox
        assert app.outbox.actions
        types = {a["action_type"] for a in app.outbox.actions}
        assert {"intercom_reply", "intercom_escalation_tag", "telegram_escalation"} == types
    finally:
        module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository = originals


def test_escalation_preserves_pipeline_escalation_reply(monkeypatch):
    import asyncio
    import app.application.support_service as module
    from app.application.support_service import SupportApplication
    from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction
    from app.domain.enums import Channel, Decision, EscalationReason, ConversationStateKind

    class FakePipeline:
        async def decide(self, message):
            return ProcessingResult(
                traces=[DecisionTrace(
                    decision=Decision.ESCALATE,
                    question=message.text,
                    escalation_reason=EscalationReason.HIGH_RISK_COMPLAINT,
                )],
                actions=[OutboundAction("reply", "Структурированный ответ оператору.", True)],
            )

    class FakeState: state = ConversationStateKind.IDLE; data = {}
    class FakeStates:
        def __init__(self, session): self.state = FakeState()
        async def get(self, entity_id, *, for_update=False): return self.state
        async def replace(self, entity_id, state, data): self.state.state = state; self.state.data = data
    class FakeEscalations:
        async def is_escalated(self, entity_id): return False
        async def record(self, **kwargs): self.recorded = kwargs
    class FakeInteraction:
        async def add_many(self, *args, **kwargs): pass
    class FakeOutbox:
        def __init__(self): self.actions=[]
        async def enqueue(self, **kwargs): self.actions.append(kwargs); return True

    originals = (module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository)
    module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository = FakeStates, lambda session: FakeInteraction(), lambda session: FakeEscalations()
    try:
        outbox = FakeOutbox()
        app = SupportApplication(FakePipeline(), object(), None, outbox)
        msg = InboundMessage("evt", "key", "topic", "conv", Channel.CONVERSATION, "жалоба")
        event = type("Event", (), {"id": 1})()
        asyncio.run(app.handle(object(), event, msg))
        assert any("Структурированный ответ оператору." in str(a.get("payload", {}).get("text", "")) for a in outbox.actions)
    finally:
        module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository = originals


def test_handle_rejects_strictly_older_provider_event(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app.application.support_service import SupportApplication
    from app.domain.models import InboundMessage
    from app.domain.enums import Channel, ConversationStateKind

    class State:
        state = ConversationStateKind.IDLE
        data = {"_ordering": {"last_event_order_at": 200}}

    class States:
        async def get(self, conversation_id, for_update=False):
            return State()

    class Escalations:
        async def is_escalated(self, conversation_id): return False

    class Session:
        async def flush(self): pass

    class Pipeline:
        async def decide(self, *args, **kwargs): raise AssertionError("stale event reached pipeline")

    monkeypatch.setattr("app.application.support_service.StateRepository", lambda session: States())
    monkeypatch.setattr("app.application.support_service.EscalatedConversationRepository", lambda session: Escalations())

    app = SupportApplication(Pipeline(), object(), None, object())
    event = SimpleNamespace(id=11, event_order_at=100)
    message = InboundMessage(
        event_id="evt-11", event_key="k", topic="conversation.user.replied", entity_id="c1",
        channel=Channel.CONVERSATION, text="old",
    )
    handled = asyncio.run(app.handle(Session(), event, message))
    assert handled is False


def test_state_replace_preserves_ordering_watermark():
    import asyncio
    from app.infrastructure.persistence.repository import StateRepository
    from app.domain.enums import ConversationStateKind

    class State:
        state = ConversationStateKind.IDLE
        data = {"_ordering": {"last_event_order_at": 123}, "candidate_ids": ["a"]}

    class Session:
        def add(self, obj): pass
        async def flush(self): pass

    repo = StateRepository(Session())
    state = State()
    async def fake_get(conversation_id):
        return state
    repo.get = fake_get
    asyncio.run(repo.replace("c1", ConversationStateKind.CHOOSING, {"candidate_ids": ["b"]}))
    assert state.data["candidate_ids"] == ["b"]
    assert state.data["_ordering"]["last_event_order_at"] == 123


def test_business_answer_is_enqueued_to_outbox(monkeypatch):
    import asyncio
    import app.application.support_service as module
    from app.application.support_service import SupportApplication
    from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction
    from app.domain.enums import Channel, Decision, ConversationStateKind

    class FakePipeline:
        async def decide(self, message, **kwargs):
            return ProcessingResult(
                traces=[DecisionTrace(
                    decision=Decision.ANSWER_LLM,
                    question=message.text,
                    kb_id="kb_016",
                    category="ЕСЛИ ВОПРОС О ДАТАХ ПОТОКОВ",
                    llm_used=True,
                )],
                actions=[OutboundAction("business_answer", "Одобренный ответ из KB.", False)],
            )

    class FakeState:
        state = ConversationStateKind.IDLE
        data = {}

    class FakeStates:
        def __init__(self, session): self.state = FakeState()
        async def get(self, entity_id, *, for_update=False): return self.state
        async def replace(self, entity_id, state, data): self.state.state = state; self.state.data = data

    class FakeEscalations:
        async def is_escalated(self, entity_id): return False

    class FakeInteraction:
        async def add_many(self, *args, **kwargs): pass

    class FakeOutbox:
        def __init__(self): self.actions = []
        async def enqueue(self, **kwargs): self.actions.append(kwargs); return True

    class FakeSession:
        async def flush(self): pass

    originals = (module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository)
    module.StateRepository = FakeStates
    module.InteractionRepository = lambda session: FakeInteraction()
    module.EscalatedConversationRepository = lambda session: FakeEscalations()
    try:
        outbox = FakeOutbox()
        app = SupportApplication(FakePipeline(), object(), None, outbox)
        msg = InboundMessage("evt", "key", "topic", "conv-answer", Channel.CONVERSATION, "какие даты потоков?")
        event = type("Event", (), {"id": 999, "event_order_at": None})()
        asyncio.run(app.handle(FakeSession(), event, msg))
        assert len(outbox.actions) == 1
        assert outbox.actions[0]["action_type"] == "intercom_reply"
        assert outbox.actions[0]["payload"]["text"] == "Одобренный ответ из KB."
    finally:
        module.StateRepository, module.InteractionRepository, module.EscalatedConversationRepository = originals
