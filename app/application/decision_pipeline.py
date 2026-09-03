import time
import logging
import hashlib
from app.domain.enums import Decision, EscalationReason
from app.domain.models import DecisionTrace, InboundMessage, OutboundAction, ProcessingResult
from app.domain.policy import escalation_ack, rephrase_prompt, oos_response, high_risk_reply
from app.domain.response_composer import ResponseComposer
from app.infrastructure.ai.knowledge_base import (
    KnowledgeBase, MatchMode, MatchResult, contains_operator_request, is_non_query_message,
)
from app.infrastructure.ai.llm_client import PolzaLLMClient
from app.core.config import settings

logger = logging.getLogger("decision_pipeline")

class DecisionPipeline:
    """Application-level policy. Retrieval and generation remain infrastructure concerns."""

    def __init__(self, kb: KnowledgeBase, llm: PolzaLLMClient | None):
        self.kb = kb
        self.llm = llm

    @staticmethod
    def _escalation_reply(text: str | None = None) -> str:
        """Compose a consistent human-handoff message."""
        return ResponseComposer.escalation_ack(text or escalation_ack())

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

    async def _decide_with_llm(
        self,
        trace: DecisionTrace,
        question: str,
        match: MatchResult,
    ) -> tuple[DecisionTrace, list[OutboundAction]]:
        """Use Polza only after deterministic routing cannot safely choose."""
        assert self.llm is not None

        candidates = list(match.relevant_entries or [])
        if not candidates:
            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.llm_used = False
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "no_candidates") + ";llm_no_candidates"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        # Keep the LLM candidate set intentionally narrow. The hybrid KB has
        # already done retrieval/ranking; the LLM is only the final semantic judge.
        candidates = candidates[:7]
        result = await self.llm.classify(question, candidates)
        trace.llm_used = True
        trace.automated = True

        if result.escalation or not result.selected_kb_ids:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = (trace.decision_reason or "supported_intent_needs_llm") + ";llm_escalation"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        business_answers: list[str] = []
        selected_ids: list[str] = []
        validation_reasons: list[str] = []
        candidate_set = {str(entry.get("id")) for entry in candidates}

        for kb_id in result.selected_kb_ids:
            kb_id = str(kb_id).strip()
            if kb_id not in candidate_set:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = "llm_selected_unknown_candidate"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            valid, reason, _ = self.kb.validate_llm_selection(question, kb_id)
            validation_reasons.append(reason)
            if not valid:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = f"llm_selection_rejected:{kb_id}:{reason}"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            entry = self.kb.entries_by_id.get(kb_id)
            entry_data = entry or {}
            requires_clarification = bool(entry_data.get("requires_clarification"))
            clarification = str(entry_data.get("clarification_question") or "").strip()
            template = str(entry_data.get("template_answer") or "").strip()

            # If the selected KB intent requires clarification, preserve that
            # contract instead of silently treating its generic template_answer
            # as a final answer. The next user turn must be routed with kb_id.
            if requires_clarification:
                if not clarification:
                    trace.decision = Decision.ESCALATE
                    trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                    trace.decision_reason = f"llm_selected_clarification_without_question:{kb_id}"
                    return trace, [OutboundAction("reply", self._escalation_reply(), True)]
                if len(result.selected_kb_ids) != 1:
                    # A single conversation state cannot safely represent two
                    # simultaneous clarification intents.
                    trace.decision = Decision.ESCALATE
                    trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                    trace.decision_reason = "llm_multi_intent_with_clarification"
                    return trace, [OutboundAction("reply", self._escalation_reply(), True)]

                trace.decision = Decision.CLARIFY
                trace.kb_id = kb_id
                trace.category = str(entry_data.get("category") or "") or trace.category
                trace.decision_reason = "llm_selected_clarification:" + ";".join(validation_reasons)
                return trace, [OutboundAction("reply", ResponseComposer.rephrase(clarification))]

            if not template:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = f"llm_selected_without_approved_text:{kb_id}"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            selected_ids.append(kb_id)
            business_answers.append(template)

        trace.decision = Decision.ANSWER_TEMPLATE
        trace.kb_id = selected_ids[0]
        first_entry = self.kb.entries_by_id.get(selected_ids[0]) or {}
        trace.category = str(first_entry.get("category") or "") or trace.category
        trace.decision_reason = "llm_selected_validated:" + ",".join(validation_reasons)
        return trace, [OutboundAction("business_answer", answer) for answer in business_answers]

    async def decide_match(self, match: MatchResult, question: str) -> tuple[DecisionTrace, list[OutboundAction]]:
        trace = self._trace(match, question)

        if match.mode == MatchMode.OPERATOR:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.USER_REQUESTED_OPERATOR
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

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
            return trace, [OutboundAction("reply", self._escalation_reply(reply), True)]

        if match.mode == MatchMode.TEMPLATE:
            trace.decision = Decision.ANSWER_TEMPLATE
            trace.automated = True
            return trace, [OutboundAction("business_answer", match.template_answer or "")]

        if match.mode == MatchMode.CLARIFICATION:
            trace.decision = Decision.CLARIFY
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(match.template_answer or rephrase_prompt()))]

        if match.mode == MatchMode.CATEGORY_CHOICE:
            # User-visible Top-3 is a first-class routing outcome. It is used only
            # for close, independently supported intents. Weak/OOS cases never
            # reach this branch; middle-band cases use MatchMode.LLM instead.
            candidates = list(match.choice_candidates or [])
            if len(candidates) < 2:
                trace.decision = Decision.ESCALATE
                trace.automated = True
                trace.llm_used = False
                trace.escalation_reason = EscalationReason.NO_CATEGORY
                trace.decision_reason = (trace.decision_reason or "choice_invalid") + ";choice_candidates_missing"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            trace.decision = Decision.CHOICE
            trace.automated = True
            trace.llm_used = False
            normalized = []
            for item in candidates[:3]:
                if not isinstance(item, dict):
                    continue
                kb_id = str(item.get("kb_id") or item.get("id") or "").strip()
                label = str(item.get("choice_label") or item.get("label") or item.get("category") or kb_id).strip()
                answer = str(item.get("template_answer") or item.get("answer") or "").strip()
                if kb_id and label:
                    normalized.append({"kb_id": kb_id, "label": label, "answer": answer})

            trace.choice_candidates = normalized
            trace.choice_candidate_ids = [x["kb_id"] for x in normalized]
            trace.decision_reason = trace.decision_reason or "ambiguous_supported_choice"
            labels = "\n".join(f"{i}. {item['label']}" for i, item in enumerate(normalized, 1))
            text = (
                "Похоже, ваш вопрос может относиться к нескольким вариантам. "
                "Выберите подходящий номер:\n\n" + labels
            )
            return trace, [OutboundAction("reply", text)]

        if match.mode == MatchMode.LLM:
            if settings.LLM_ENABLED and self.llm and match.relevant_entries:
                return await self._decide_with_llm(trace, question, match)

            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.llm_used = False
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "supported_intent_needs_llm") + ";llm_disabled_escalate"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

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

        if match.mode == MatchMode.NO_MATCH and (
            match.decision_reason.startswith("oos:")
            or match.decision_reason in {"oos", "oos:no_domain_support", "oos:no_supported_intent"}
        ):
            # A substantive message that falls outside the supported intent set
            # is not user-visible noise. Escalate instead of silently dropping it.
            # Genuine noise is filtered in decide() before the KB is queried.
            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "oos") + ";unsupported_question_escalate"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        if match.mode == MatchMode.NO_MATCH:
            # Any non-noise substantive query that reaches NO_MATCH must have an
            # explicit user-facing fallback. Do not silently abandon it.
            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "no_match") + ";no_safe_answer_escalate"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        trace.decision = Decision.ESCALATE
        trace.escalation_reason = EscalationReason.NO_CATEGORY
        return trace, [OutboundAction("reply", self._escalation_reply(), True)]

    async def decide(self, message: InboundMessage) -> ProcessingResult:
        started = time.perf_counter()

        raw_text = message.text or ""
        logger.info(
            "inbound_query | entity_id=%s | event_id=%s | channel=%s | text_len=%d | text_sha256=%s | text=%r",
            getattr(message, "entity_id", None),
            getattr(message, "event_id", None),
            getattr(message, "channel", None),
            len(raw_text),
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
            raw_text,
        )

        # Operator request is an explicit user action and must survive the
        # generic one-word gate (e.g. a standalone "оператор").
        if contains_operator_request(message.text):
            trace = DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="user_requested_operator",
                escalation_reason=EscalationReason.USER_REQUESTED_OPERATOR,
            )
            result = ProcessingResult(
                traces=[trace],
                actions=[OutboundAction("reply", self._escalation_reply(), True)],
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # Ignore generic acknowledgements/greetings and standalone one-word
        # messages before expensive routing. Contextual one-word replies are
        # consumed by SupportApplication state handlers before this method.
        if is_non_query_message(message.text):
            trace = DecisionTrace(
                decision=Decision.IGNORE,
                question=message.text,
                decision_reason="noise:non_query_or_single_word",
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
                actions=[OutboundAction("reply", self._escalation_reply(), True)],
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        matches = self.kb.match_many(message.text) or [self.kb.match(message.text)]
        traces: list[DecisionTrace] = []
        raw_actions: list[OutboundAction] = []

        for match in matches[:5]:
            question = match.subquestion or message.text
            trace, actions = await self.decide_match(match, question)
            question_text = question or ""
            logger.info(
                "routing_decision | question_len=%d | question_sha256=%s | question=%r | mode=%s | decision=%s | kb_id=%s | score=%.4f | margin=%.4f | reason=%s",
                len(question_text),
                hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:16],
                question_text,
                getattr(match.mode, "value", match.mode),
                getattr(trace.decision, "value", trace.decision),
                trace.kb_id,
                float(trace.score or 0.0),
                float(trace.margin or 0.0),
                trace.decision_reason or "",
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
            actions.append(OutboundAction("reply", self._escalation_reply(), True))

        result = ProcessingResult(traces=traces, actions=actions)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

