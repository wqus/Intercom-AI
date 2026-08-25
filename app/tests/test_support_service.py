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
