import time
import logging
from app.domain.enums import Decision, EscalationReason
from app.domain.models import DecisionTrace, InboundMessage, OutboundAction, ProcessingResult
from app.domain.policy import choice_prompt, escalation_ack, rephrase_prompt
from app.infrastructure.ai.knowledge_base import KnowledgeBase, MatchMode, MatchResult, contains_operator_request
from app.infrastructure.ai.llm_client import YandexLLMClient
from app.core.config import settings

logger = logging.getLogger("decision_pipeline")

class DecisionPipeline:
    """Application-level policy. Retrieval and generation remain infrastructure concerns."""

    def __init__(self, kb: KnowledgeBase, llm: YandexLLMClient):
        self.kb = kb
        self.llm = llm

    def _trace(self, match: MatchResult, question: str) -> DecisionTrace:
        return DecisionTrace(
            decision=Decision.ESCALATE,
            question=question,
            kb_id=match.kb_id,
            contrastive_margin=getattr(match, "contrastive_margin", 0.0),
            contrastive_competitor_id=getattr(match, "contrastive_competitor_id", None),
            contrastive_reason=getattr(match, "contrastive_reason", ""),
            category=match.category,
            score=match.score,
            margin=match.margin,
            embedding_score=match.embedding_score,
            intent_fit_score=match.intent_fit_score,
            family_fit_score=match.family_fit_score,
            query_quality_score=match.query_quality_score,
            decision_reason=match.decision_reason,
        )

    async def decide_match(self, match: MatchResult, question: str) -> tuple[DecisionTrace, list[OutboundAction]]:
        trace = self._trace(match, question)

        if match.mode == MatchMode.OPERATOR:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.USER_REQUESTED_OPERATOR
            return trace, [OutboundAction("reply", escalation_ack(), True)]

        if match.mode == MatchMode.TEMPLATE:
            trace.decision = Decision.ANSWER_TEMPLATE
            trace.automated = True
            return trace, [OutboundAction("reply", match.template_answer)]

        if match.mode == MatchMode.CLARIFICATION:
            trace.decision = Decision.CLARIFY
            trace.automated = True
            return trace, [OutboundAction("reply", match.template_answer)]

        if match.mode == MatchMode.CATEGORY_CHOICE:
            labels = [
                str(c.get("choice_label") or c.get("category") or c.get("kb_id"))
                for c in (match.choice_candidates or [])
            ]
            trace.decision = Decision.CHOICE
            trace.automated = True
            # SAFE ROLLOUT: Top-3/category-choice is intentionally NOT exposed
            # to the customer while the LLM/routing quality is under validation.
            if not settings.USER_VISIBLE_TOP3:
                trace.decision_reason = (trace.decision_reason or "") + ";top3_user_output_disabled"
                return trace, []
            return trace, [OutboundAction("reply", choice_prompt(labels))]

        if match.mode == MatchMode.LLM and match.relevant_entries:
            # CONSERVATIVE PRODUCTION MODE:
            # Never call the LLM from the live support path. A case that would
            # normally be delegated to the model is converted into a user-visible
            # Top-3 choice. This is intentionally conservative: user choice is
            # safer than a model guess.
            candidates = list(match.choice_candidates or [])
            if len(candidates) < 2:
                try:
                    candidates = self.kb.get_category_choice_candidates(question, top_n=3)
                except Exception:
                    logger.exception("Failed to build Top-3 fallback for LLM route")
                    candidates = []

            if len(candidates) >= 2:
                labels = [
                    str(c.get("choice_label") or c.get("category") or c.get("kb_id"))
                    for c in candidates[:3]
                ]
                trace.decision = Decision.CHOICE
                trace.automated = True
                trace.llm_used = False
                trace.decision_reason = (
                    trace.decision_reason or "supported_intent_needs_llm"
                ) + ";llm_disabled_top3_fallback"
                return trace, [OutboundAction("reply", choice_prompt(labels))]

            # No safe Top-3 means we still must not call the LLM. Ask for a
            # better-formed question instead of guessing a category.
            trace.decision = Decision.REPHRASE
            trace.automated = True
            trace.llm_used = False
            trace.decision_reason = (
                trace.decision_reason or "supported_intent_needs_llm"
            ) + ";llm_disabled_no_safe_top3"
            return trace, [OutboundAction("reply", rephrase_prompt())]

            # FUTURE LLM PATH (intentionally disabled for now):
            # result = await self.llm.classify(question, match.relevant_entries)
            # ... validate selected_kb_id with self.kb.validate_llm_selection(...)
            # ... send result.answer only after the model is re-enabled and audited.

        if match.mode == MatchMode.REPHRASE:
            trace.decision = Decision.REPHRASE
            trace.automated = True
            return trace, [OutboundAction("reply", rephrase_prompt())]

        if match.mode == MatchMode.NO_MATCH and match.decision_reason.startswith("noise:"):
            # Noise is a routing outcome, not a support failure. Do not send
            # the customer a fallback/ escalation acknowledgement.
            trace.decision = Decision.IGNORE
            trace.automated = True
            return trace, []

        if match.mode == MatchMode.NO_MATCH and match.decision_reason in {
            "insufficient_context", "low_confidence", "unsupported_or_ambiguous"
        }:
            # Unknown/ambiguous intent is not automatically a human-ticket.
            # First try to recover the missing discriminator from the user.
            # Escalation should be reserved for explicit operator requests,
            # unsupported/high-risk cases, or a second failed clarification.
            trace.decision = Decision.REPHRASE
            trace.automated = True
            return trace, [OutboundAction("reply", rephrase_prompt())]

        trace.decision = Decision.ESCALATE
        trace.escalation_reason = EscalationReason.NO_CATEGORY
        return trace, [OutboundAction("reply", escalation_ack(), True)]

    async def decide(self, message: InboundMessage) -> ProcessingResult:
        started = time.perf_counter()

        if message.has_attachments:
            trace = DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="attachments",
                escalation_reason=EscalationReason.ATTACHMENTS,
            )
            result = ProcessingResult(
                traces=[trace],
                actions=[OutboundAction("reply", escalation_ack(), True)],
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        if contains_operator_request(message.text):
            trace = DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="user_requested_operator",
                escalation_reason=EscalationReason.USER_REQUESTED_OPERATOR,
            )
            result = ProcessingResult(
                traces=[trace],
                actions=[OutboundAction("reply", escalation_ack(), True)],
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        matches = self.kb.match_many(message.text) or [self.kb.match(message.text)]
        traces: list[DecisionTrace] = []
        raw_actions: list[OutboundAction] = []

        for match in matches[:5]:
            question = match.subquestion or message.text
            trace, actions = await self.decide_match(match, question)
            traces.append(trace)
            raw_actions.extend(actions)

        replies = list(dict.fromkeys(
            action.text for action in raw_actions
            if action.kind == "reply" and action.text
        ))
        needs_escalation = any(action.escalate for action in raw_actions)

        actions: list[OutboundAction] = []
        if replies:
            actions.append(OutboundAction("reply", "\n\n".join(replies), needs_escalation))
        elif needs_escalation:
            actions.append(OutboundAction("reply", escalation_ack(), True))

        result = ProcessingResult(traces=traces, actions=actions)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

