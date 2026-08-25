from enum import Enum

class Channel(str, Enum):
    CONVERSATION = "conversation"
    TICKET = "ticket"

class Decision(str, Enum):
    ANSWER_TEMPLATE = "answer_template"
    ANSWER_LLM = "answer_llm"
    CLARIFY = "clarify"
    CHOICE = "choice"
    REPHRASE = "rephrase"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    OOS = "oos"

class EventStatus(str, Enum):
    NEW = "new"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

class ConversationStateKind(str, Enum):
    IDLE = "idle"
    CHOOSING = "choosing"
    CLARIFYING = "clarifying"
    REPHRASING = "rephrasing"
    ESCALATED = "escalated"

class EscalationReason(str, Enum):
    USER_REQUESTED_OPERATOR = "user_requested_operator"
    NO_CATEGORY = "no_category"
    LLM_CANNOT_ANSWER = "llm_cannot_answer"
    INTERNAL_ERROR = "internal_error"
    WANTS_NEW_SCHOOL = "wants_new_school"
    ATTACHMENTS = "attachments"
    DELIVERY_FAILURE = "delivery_failure"
    CLARIFICATION_EXHAUSTED = "clarification_exhausted"

