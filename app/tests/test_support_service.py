import asyncio

from app.application.support_service import SupportApplication
from app.domain.enums import Channel, Decision, EscalationReason
from app.domain.models import InboundMessage, ProcessingResult, DecisionTrace, OutboundAction


def test_processing_result_escalation_state_is_derived_from_action():
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

    class FakeIntercom:
        def __init__(self):
            self.replies = 0
            self.tags = 0
        async def send_reply(self, *args):
            self.replies += 1
        async def add_escalation_tag(self, *args):
            self.tags += 1

    class FakeTelegram:
        def __init__(self):
            self.notifications = 0
        async def notify_escalation(self, **kwargs):
            self.notifications += 1
            return True

    class FakeState:
        def __init__(self):
            from app.domain.enums import ConversationStateKind
            self.state = ConversationStateKind.IDLE
            self.data = {}

    class FakeStates:
        def __init__(self, session):
            self.state = FakeState()
            self.replaced = []
        async def get(self, entity_id):
            return self.state
        async def replace(self, entity_id, state, data):
            self.replaced.append((entity_id, state, data))

    # Patch the repository classes only inside this regression test.
    import app.application.support_service as module
    original_state = module.StateRepository
    original_interaction = module.InteractionRepository
    module.StateRepository = FakeStates
    module.InteractionRepository = lambda session: type(
        "FakeInteractionRepo", (), {"add_many": lambda self, *args, **kwargs: asyncio.sleep(0)}
    )()
    try:
        intercom = FakeIntercom()
        telegram = FakeTelegram()
        app = SupportApplication(FakePipeline(), intercom, telegram)
        message = InboundMessage("evt", "key", "conversation.user.replied", "conv", Channel.CONVERSATION, "оператор")
        asyncio.run(app.handle(object(), type("Event", (), {"id": 1})(), message))
        assert intercom.replies == 1
        assert telegram.notifications == 1
        assert intercom.tags == 0
    finally:
        module.StateRepository = original_state
        module.InteractionRepository = original_interaction
