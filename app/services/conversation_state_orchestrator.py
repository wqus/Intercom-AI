from __future__ import annotations

from dataclasses import dataclass

from app.models.conversation_category_choice import ConversationCategoryChoice
from app.models.conversation_rephrase_state import ConversationRephraseState
from app.services.knowledge_base import KnowledgeBase, MatchMode, MatchResult


MAX_REPHRASE_FAILURES = 2
MAX_CATEGORY_CHOICE_ATTEMPTS = 2


@dataclass(slots=True)
class PendingChoiceResolution:
    candidate_ids: list[str]


def candidate_ids_from_state(state: ConversationCategoryChoice) -> list[str]:
    result = [state.candidate_1, state.candidate_2]
    if state.candidate_3:
        result.append(state.candidate_3)
    return result


def selected_candidate_ids_from_text(
    kb: KnowledgeBase,
    state: ConversationCategoryChoice,
    text: str,
) -> list[str]:
    candidate_ids = candidate_ids_from_state(state)
    indexes = kb.parse_category_choice(text, max_index=len(candidate_ids))
    if not indexes:
        return []

    return [candidate_ids[index - 1] for index in indexes]


def should_escalate_after_rephrase_failure(
    state: ConversationRephraseState,
) -> bool:
    return state.attempts >= MAX_REPHRASE_FAILURES


def should_cancel_bad_choice_attempt(
    state: ConversationCategoryChoice,
) -> bool:
    return state.attempts >= MAX_CATEGORY_CHOICE_ATTEMPTS
