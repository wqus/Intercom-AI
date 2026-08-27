import time
import logging
from app.domain.enums import Decision, EscalationReason
from app.domain.models import DecisionTrace, InboundMessage, OutboundAction, ProcessingResult
from app.domain.policy import escalation_ack, rephrase_prompt, oos_response, high_risk_reply
from app.domain.response_composer import ResponseComposer
from app.infrastructure.ai.knowledge_base import (
    KnowledgeBase, MatchMode, MatchResult, contains_operator_request, is_non_query_message,
)
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
            semantic_score=getattr(match, "semantic_score", 0.0),
            lexical_score=getattr(match, "lexical_score", 0.0),
            fuzzy_score=getattr(match, "fuzzy_score", 0.0),
            evidence_score=getattr(match, "evidence_score", 0.0),
            real_intent_evidence=getattr(match, "real_intent_evidence", 0.0),
            domain_support_score=getattr(match, "domain_support_score", 0.0),
            ontology_alignment_score=getattr(match, "ontology_alignment_score", 0.0),
            negative_contrast_score=getattr(match, "negative_contrast_score", 0.0),
            decision_reason=match.decision_reason,
        )

    async def decide_match(self, match: MatchResult, question: str) -> tuple[DecisionTrace, list[OutboundAction]]:
        trace = self._trace(match, question)

        if match.mode == MatchMode.OPERATOR:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.USER_REQUESTED_OPERATOR
            return trace, [OutboundAction("reply", escalation_ack(), True)]

        # High-risk: refund / personal_data / complaint — escalate with data collection.
        if match.decision_reason.startswith("high_risk:"):
            kind = match.decision_reason.split(":", 1)[-1]
            reason_map = {
                "refund": EscalationReason.HIGH_RISK_REFUND,
                "personal_data": EscalationReason.HIGH_RISK_PERSONAL_DATA,
                "complaint": EscalationReason.HIGH_RISK_COMPLAINT,
            }
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = reason_map.get(kind, EscalationReason.NO_CATEGORY)
            trace.decision_reason = match.decision_reason
            reply = high_risk_reply(kind)
            return trace, [OutboundAction("reply", ResponseComposer.escalation_ack(reply), True)]

        if match.mode == MatchMode.TEMPLATE:
            trace.decision = Decision.ANSWER_TEMPLATE
            trace.automated = True
            return trace, [OutboundAction("business_answer", match.template_answer or "")]

        if match.mode == MatchMode.CLARIFICATION:
            trace.decision = Decision.CLARIFY
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(match.template_answer or rephrase_prompt()))]

        if match.mode == MatchMode.CATEGORY_CHOICE:
            cands = list(match.choice_candidates or []) or list(getattr(match, "top_candidates", None) or [])
            labels = []
            ids = []
            for c in cands[:3]:
                if not isinstance(c, dict):
                    continue
                label = str(c.get("choice_label") or c.get("category") or c.get("kb_id") or "")
                cid = str(c.get("kb_id") or "")
                if not label:
                    continue
                labels.append(label)
                if cid:
                    ids.append(cid)
            trace.decision = Decision.CHOICE
            trace.automated = True
            trace.choice_candidate_ids = ids if len(ids) >= 2 else None
            if not settings.USER_VISIBLE_TOP3:
                trace.decision_reason = (trace.decision_reason or "") + ";top3_user_output_disabled"
                return trace, []
            if len(labels) < 2:
                trace.decision = Decision.REPHRASE
                return trace, [OutboundAction("reply", ResponseComposer.rephrase(rephrase_prompt()))]
            return trace, [OutboundAction("reply", ResponseComposer.choice(labels))]

        if match.mode == MatchMode.LLM and match.relevant_entries:
            # LLM disabled in live: always prefer Top-3 over rephrase.
            candidates = list(match.choice_candidates or [])
            if len(candidates) < 2:
                candidates = list(getattr(match, "top_candidates", None) or [])
            if len(candidates) < 2:
                try:
                    candidates = self.kb.get_category_choice_candidates(question, top_n=3)
                except Exception:
                    logger.exception("Failed to build Top-3 fallback for LLM route")
                    candidates = []

            if len(candidates) >= 2:
                labels = []
                ids = []
                for c in candidates[:3]:
                    if not isinstance(c, dict):
                        continue
                    label = str(c.get("choice_label") or c.get("category") or c.get("kb_id") or "")
                    cid = str(c.get("kb_id") or "")
                    if not label:
                        continue
                    labels.append(label)
                    if cid:
                        ids.append(cid)
                if len(labels) >= 2:
                    trace.decision = Decision.CHOICE
                    trace.automated = True
                    trace.llm_used = False
                    trace.choice_candidate_ids = ids if len(ids) >= 2 else None
                    trace.decision_reason = (
                        trace.decision_reason or "supported_intent_needs_llm"
                    ) + ";llm_disabled_top3_fallback"
                    return trace, [OutboundAction("reply", ResponseComposer.choice(labels))]

            # Last resort only when we truly have nothing to offer.
            trace.decision = Decision.REPHRASE
            trace.automated = True
            trace.llm_used = False
            trace.decision_reason = (
                trace.decision_reason or "supported_intent_needs_llm"
            ) + ";llm_disabled_no_safe_top3"
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(rephrase_prompt()))]

            # FUTURE LLM PATH (intentionally disabled for now):
            # result = await self.llm.classify(question, match.relevant_entries)
            # ... validate selected_kb_id with self.kb.validate_llm_selection(...)
            # ... send result.answer only after the model is re-enabled and audited.

        if match.mode == MatchMode.REPHRASE:
            trace.decision = Decision.REPHRASE
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(rephrase_prompt()))]

        if match.mode == MatchMode.NO_MATCH and match.decision_reason.startswith("noise:"):
            # Noise is a routing outcome, not a support failure. Do not send
            # the customer a fallback/ escalation acknowledgement.
            trace.decision = Decision.IGNORE
            trace.automated = True
            return trace, []

        if match.mode == MatchMode.NO_MATCH and match.decision_reason in {
            "oos", "oos:external_target", "oos:no_domain_support", "oos:no_supported_intent"
        }:
            trace.decision = Decision.OOS
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.oos(oos_response()))]

        if match.mode == MatchMode.NO_MATCH and match.decision_reason in {
            "insufficient_context", "low_confidence", "unsupported_or_ambiguous",
            "unsupported_object", "object_mismatch", "facet_mismatch",
        }:
            # Prefer Top-3 over generic rephrase when ranked candidates exist.
            candidates = list(getattr(match, "top_candidates", None) or [])
            labels = [
                str(c.get("choice_label") or c.get("category") or c.get("kb_id"))
                for c in candidates[:3]
                if isinstance(c, dict) and (c.get("choice_label") or c.get("category") or c.get("kb_id"))
            ]
            if len(labels) >= 2 and float(match.score or 0.0) >= 0.35:
                ids = [
                    str(c.get("kb_id"))
                    for c in candidates[:3]
                    if isinstance(c, dict) and c.get("kb_id")
                ]
                trace.decision = Decision.CHOICE
                trace.automated = True
                trace.choice_candidate_ids = ids if len(ids) >= 2 else None
                trace.decision_reason = (match.decision_reason or "") + ";top3_from_no_match"
                return trace, [OutboundAction("reply", ResponseComposer.choice(labels))]
            trace.decision = Decision.REPHRASE
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(rephrase_prompt()))]

        trace.decision = Decision.ESCALATE
        trace.escalation_reason = EscalationReason.NO_CATEGORY
        return trace, [OutboundAction("reply", escalation_ack(), True)]

    async def decide(self, message: InboundMessage) -> ProcessingResult:
        started = time.perf_counter()

        logger.info(
            "inbound_query | entity_id=%s | event_id=%s | channel=%s | text=%s",
            getattr(message, "entity_id", None),
            getattr(message, "event_id", None),
            getattr(message, "channel", None),
            (message.text or "")[:2000],
        )

        # Ignore generic acknowledgements/greetings before attachment escalation.
        # A file does not by itself turn "Спасибо" into a support ticket.
        if is_non_query_message(message.text):
            trace = DecisionTrace(
                decision=Decision.IGNORE,
                question=message.text,
                decision_reason="noise:non_query_exact",
                automated=True,
            )
            result = ProcessingResult(traces=[trace], actions=[])
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

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
            logger.info(
                "routing_decision | question=%s | mode=%s | decision=%s | kb_id=%s | score=%.4f | margin=%.4f | reason=%s",
                (question or "")[:400],
                getattr(match.mode, "value", match.mode),
                getattr(trace.decision, "value", trace.decision),
                trace.kb_id,
                float(trace.score or 0.0),
                float(trace.margin or 0.0),
                (trace.decision_reason or "")[:200],
            )
            traces.append(trace)
            raw_actions.extend(actions)

        business_answers = list(dict.fromkeys(
            action.text for action in raw_actions
            if action.kind == "business_answer" and action.text
        ))
        replies = list(dict.fromkeys(
            action.text for action in raw_actions
            if action.kind == "reply" and action.text
        ))
        needs_escalation = any(action.escalate for action in raw_actions)

        actions: list[OutboundAction] = []
        if business_answers and not replies:
            actions.append(OutboundAction(
                "reply",
                ResponseComposer.final_answer(business_answers),
                needs_escalation,
            ))
        elif business_answers:
            # Mixed multi-intent outcome: keep the business answers together and
            # do not append the final footer until the interaction itself is over.
            actions.append(OutboundAction(
                "reply",
                ResponseComposer.final_answer(business_answers, include_footer=False),
                needs_escalation,
            ))
            actions.append(OutboundAction("reply", "\n\n".join(replies), needs_escalation))
        elif replies:
            actions.append(OutboundAction("reply", "\n\n".join(replies), needs_escalation))
        elif needs_escalation:
            actions.append(OutboundAction("reply", ResponseComposer.escalation_ack(escalation_ack()), True))

        result = ProcessingResult(traces=traces, actions=actions)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

