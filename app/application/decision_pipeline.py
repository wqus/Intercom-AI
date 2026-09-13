import time
import logging
import hashlib
from app.domain.enums import Decision, EscalationReason
from app.domain.models import DecisionTrace, InboundMessage, OutboundAction, ProcessingResult
from app.domain.policy import escalation_ack, rephrase_prompt, high_risk_reply
from app.domain.response_composer import ResponseComposer
from app.infrastructure.ai.knowledge_base import (
    KnowledgeBase, MatchMode, MatchResult, contains_operator_request, detect_high_risk,
    is_non_query_message,
)
from app.infrastructure.ai.llm_client import LLMResult, PolzaLLMClient
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
        conversation_context: list[dict[str, str]] | None = None,
        *,
        taxonomy_mode: bool = False,
    ) -> tuple[DecisionTrace, list[OutboundAction]]:
        """Use the semantic model as the primary router after deterministic policy guards."""
        assert self.llm is not None

        candidates = list(match.relevant_entries or [])
        if not candidates:
            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.llm_used = False
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "no_candidates") + ";llm_no_candidates"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        # Legacy retrieval mode keeps the candidate set narrow. AI-first taxonomy
        # mode intentionally sends the compact full intent directory (about 40–50
        # intents), because the model is the classifier rather than a reranker.
        if not taxonomy_mode:
            candidates = candidates[:10]
        llm_started = time.perf_counter()
        logger.info(
            "llm_dispatch | taxonomy=%s | question_len=%d | question_sha256=%s | "
            "catalog_size=%d | context_messages=%d",
            taxonomy_mode,
            len(question),
            hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            len(candidates),
            len(conversation_context or []),
        )
        try:
            if taxonomy_mode:
                result = await self.llm.classify(
                    question, candidates, conversation_context, taxonomy_mode=True
                )
            else:
                result = await self.llm.classify(question, candidates, conversation_context)
        except TypeError as exc:
            # Keep small test doubles / legacy adapters compatible while the
            # production client supports conversation context.
            if "positional" not in str(exc) and "argument" not in str(exc):
                result = LLMResult([], unavailable=True, reason="llm_request_error", decision="escalate")
            else:
                try:
                    result = await self.llm.classify(question, candidates)
                except Exception as legacy_exc:
                    logger.exception("LLM legacy request failed: %s", legacy_exc)
                    result = LLMResult([], unavailable=True, reason="llm_request_error", decision="escalate")
        except Exception as exc:
            logger.exception("LLM request failed: %s", exc)
            result = LLMResult([], unavailable=True, reason="llm_request_error", decision="escalate")
        llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000
        trace.llm_used = True
        trace.automated = True
        logger.info(
            "llm_router_result | taxonomy=%s | elapsed_ms=%.1f | decision=%s | "
            "selected_kb_ids=%s | certainty=%s | compound=%s | unavailable=%s | escalation=%s | reason=%s",
            taxonomy_mode,
            llm_elapsed_ms,
            getattr(result, "decision", None),
            list(getattr(result, "selected_kb_ids", []) or []),
            getattr(result, "certainty", None),
            getattr(result, "is_compound", None),
            bool(getattr(result, "unavailable", False)),
            bool(getattr(result, "escalation", False)),
            getattr(result, "reason", None),
        )

        if result.unavailable:
            # AI-first taxonomy mode has no semantic fallback: when the model is
            # unavailable or malformed, hand the conversation to an operator.
            # Legacy retrieval mode keeps the old Top-3 technical fallback for
            # backwards compatibility behind its feature flag.
            if not taxonomy_mode:
                fallback = list(match.top_candidates or [])[:3]
                labels = [
                    str(item.get("choice_label") or item.get("label") or item.get("category") or item.get("id") or "").strip()
                    for item in fallback
                ]
                if len(fallback) == 3 and all(labels):
                    trace.decision = Decision.CHOICE
                    trace.choice_candidates = [
                        {
                            "kb_id": str(item.get("id")),
                            "label": label,
                            "answer": str(item.get("template_answer") or "").strip(),
                        }
                        for item, label in zip(fallback, labels)
                    ]
                    trace.choice_candidate_ids = [str(item.get("id")) for item in fallback]
                    trace.escalation_reason = None
                    trace.decision_reason = (trace.decision_reason or "ai_routing") + ";llm_error_top3_fallback"
                    return trace, [OutboundAction("reply", ResponseComposer.choice(labels))]

            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = (trace.decision_reason or "ai_routing") + ";llm_unavailable"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        if result.decision == "ignore":
            if taxonomy_mode and str(getattr(result, "certainty", "")).lower() != "high":
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = "llm_ignore_requires_high_certainty"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]
            trace.decision = Decision.IGNORE
            trace.decision_reason = (trace.decision_reason or "ai_routing") + ";llm_ignore"
            return trace, []

        if result.decision == "escalate":
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = (trace.decision_reason or "supported_intent_needs_llm") + ";llm_escalation"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        # Validate the model's structured contract before touching KB/state.
        # JSON-schema constrains the provider, but application-level invariants
        # must still be enforced for mocks, legacy adapters and provider quirks.
        selected_ids = [str(x).strip() for x in (result.selected_kb_ids or []) if str(x).strip()]
        if taxonomy_mode and result.decision == "answer" and len(selected_ids) > 1 and getattr(result, "is_compound", None) is not True:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_multiple_answers_without_compound_intent"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if taxonomy_mode and result.decision == "answer" and str(getattr(result, "certainty", "")).lower() != "high":
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_answer_requires_high_certainty"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if taxonomy_mode and result.decision == "clarify" and str(getattr(result, "certainty", "")).lower() == "low":
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_low_certainty_cannot_clarify"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if taxonomy_mode and result.decision == "ignore" and str(getattr(result, "certainty", "")).lower() != "high":
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_ignore_requires_high_certainty"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if result.decision == "choice" and not 2 <= len(selected_ids) <= 3:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_invalid_choice_cardinality"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if result.decision == "clarify" and len(selected_ids) != 1:
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_invalid_clarify_cardinality"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if result.decision == "answer":
            valid_answer_cardinality = (
                1 <= len(selected_ids) <= 4
                if taxonomy_mode
                else 1 <= len(selected_ids) <= 3
            )
            if not valid_answer_cardinality:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = "llm_invalid_answer_cardinality"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        if len(selected_ids) != len(set(selected_ids)):
            trace.decision = Decision.ESCALATE
            trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
            trace.decision_reason = "llm_duplicate_candidate_ids"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]
        # Keep downstream loops tied to the normalized contract.
        result.selected_kb_ids = selected_ids
        trace.selected_kb_ids = list(selected_ids)

        business_answers: list[str] = []
        selected_ids: list[str] = []
        candidate_set = {str(entry.get("id")) for entry in candidates}

        for kb_id in result.selected_kb_ids:
            kb_id = str(kb_id).strip()
            if kb_id not in candidate_set:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = "llm_selected_unknown_candidate"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            entry = self.kb.entries_by_id.get(kb_id)
            entry_data = entry or {}
            requires_clarification = bool(entry_data.get("requires_clarification"))
            clarification = str(entry_data.get("clarification_question") or "").strip()
            template = str(entry_data.get("template_answer") or "").strip()

            # The model decides whether the current turn actually lacks the
            # discriminator needed by a conditional intent. `requires_clarification`
            # describes that the category has a clarification branch; it must not
            # force an unnecessary question when the user already supplied the
            # distinguishing fact.
            if result.decision == "clarify":
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
                trace.decision_reason = "llm_selected_clarification"
                return trace, [OutboundAction("reply", ResponseComposer.rephrase(clarification))]

            if not template:
                trace.decision = Decision.ESCALATE
                trace.escalation_reason = EscalationReason.LLM_CANNOT_ANSWER
                trace.decision_reason = f"llm_selected_without_approved_text:{kb_id}"
                return trace, [OutboundAction("reply", self._escalation_reply(), True)]

            selected_ids.append(kb_id)
            business_answers.append(template)

        trace.selected_kb_ids = list(selected_ids)
        trace.decision = Decision.ANSWER_LLM
        trace.kb_id = selected_ids[0]
        first_entry = self.kb.entries_by_id.get(selected_ids[0]) or {}
        trace.category = str(first_entry.get("category") or "") or trace.category
        trace.decision_reason = (
            "llm_selected_from_full_taxonomy" if taxonomy_mode
            else "llm_selected_from_retrieved_candidates"
        )
        return trace, [OutboundAction("business_answer", answer) for answer in business_answers]

    async def decide_match(
        self, match: MatchResult, question: str,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> tuple[DecisionTrace, list[OutboundAction]]:
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
            trace.selected_kb_ids = [str(match.kb_id)] if match.kb_id else None
            return trace, [OutboundAction("business_answer", match.template_answer or "")]

        if match.mode == MatchMode.CLARIFICATION:
            trace.decision = Decision.CLARIFY
            trace.automated = True
            return trace, [OutboundAction("reply", ResponseComposer.rephrase(match.template_answer or rephrase_prompt()))]

        if match.mode == MatchMode.CATEGORY_CHOICE:
            # Legacy compatibility: retrieval must never emit a user-facing
            # Top-3 by itself. Route the neighborhood through the LLM; the
            # application creates Top-3 only when that LLM call fails.
            if settings.LLM_ENABLED and self.llm and match.relevant_entries:
                match.mode = MatchMode.LLM
                return await self._decide_with_llm(trace, question, match, conversation_context)
            trace.decision = Decision.ESCALATE
            trace.automated = True
            trace.llm_used = False
            trace.escalation_reason = EscalationReason.NO_CATEGORY
            trace.decision_reason = (trace.decision_reason or "choice_requires_llm") + ";llm_disabled_escalate"
            return trace, [OutboundAction("reply", self._escalation_reply(), True)]

        if match.mode == MatchMode.LLM:
            if settings.LLM_ENABLED and self.llm and match.relevant_entries:
                return await self._decide_with_llm(trace, question, match, conversation_context)

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

    async def decide(
        self,
        message: InboundMessage,
        *,
        is_first_message: bool = True,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> ProcessingResult:
        started = time.perf_counter()

        raw_text = message.text or ""
        log_values = (
            getattr(message, "entity_id", None),
            getattr(message, "event_id", None),
            getattr(message, "channel", None),
            len(raw_text),
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
        )
        if settings.LOG_SENSITIVE_CONTENT:
            logger.info(
                "inbound_query | entity_id=%s | event_id=%s | channel=%s | text_len=%d | text_sha256=%s | text=%r",
                *log_values, raw_text,
            )
        else:
            logger.info(
                "inbound_query | entity_id=%s | event_id=%s | channel=%s | text_len=%d | text_sha256=%s",
                *log_values,
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

        # Ignore is deliberately tiny: only a curated acknowledgement/noise
        # phrase, only after the conversation already contains a user turn.
        # Never short-circuit a pending clarification/menu context: a word such as
        # "ок" or "да" may be the user's substantive response to the last bot turn.
        has_pending_context = any(
            isinstance(item, dict)
            and str(item.get("role") or "").strip().lower() == "system"
            and str(item.get("text") or "").startswith("PENDING_")
            for item in (conversation_context or [])
        )
        if is_non_query_message(message.text) and not has_pending_context:
            if not is_first_message:
                trace = DecisionTrace(
                    decision=Decision.IGNORE,
                    question=message.text,
                    decision_reason="noise:contextual_non_query",
                    automated=True,
                )
                result = ProcessingResult(traces=[trace], actions=[])
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

            # The first turn is never silently ignored. A greeting/acknowledgement
            # without an actual support request cannot be answered from KB, so the
            # safe policy is a human hand-off rather than a generic clarification.
            trace = DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="first_turn_non_query_escalate",
                escalation_reason=EscalationReason.NO_CATEGORY,
                automated=True,
            )
            result = ProcessingResult(
                traces=[trace],
                actions=[OutboundAction("reply", self._escalation_reply(), True)],
            )
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

        # AI-first router: classify the current turn against the compact full
        # taxonomy. This is the primary path when enabled. Deterministic high-risk
        # patterns still bypass the model because those are policy, not semantics.
        taxonomy_mode = bool(
            getattr(settings, "LLM_ROUTER_MODE", "taxonomy").lower() == "taxonomy"
            and settings.LLM_ENABLED
            and self.llm
            and hasattr(self.kb, "intent_catalog_for_llm")
        )
        if taxonomy_mode:
            high_risk = detect_high_risk(message.text)
            if high_risk:
                reason_map = {
                    "refund": EscalationReason.HIGH_RISK_REFUND,
                    "personal_data": EscalationReason.HIGH_RISK_PERSONAL_DATA,
                    "complaint": EscalationReason.HIGH_RISK_COMPLAINT,
                }
                trace = DecisionTrace(
                    decision=Decision.ESCALATE,
                    question=message.text,
                    decision_reason=f"high_risk:{high_risk}",
                    escalation_reason=reason_map.get(high_risk, EscalationReason.NO_CATEGORY),
                )
                reply = high_risk_reply(high_risk)
                result = ProcessingResult(
                    traces=[trace],
                    actions=[OutboundAction("reply", self._escalation_reply(reply), True)],
                )
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result

            catalog = self.kb.intent_catalog_for_llm()
            match = MatchResult(
                mode=MatchMode.LLM,
                score=0.0,
                category=None,
                template_answer=None,
                kb_id=None,
                relevant_entries=catalog,
                top_candidates=None,
                subquestion=message.text,
                decision_reason="ai_first_full_taxonomy",
            )
            trace, actions = await self._decide_with_llm(
                self._trace(match, message.text),
                message.text,
                match,
                conversation_context,
                taxonomy_mode=True,
            )
            traces = [trace]
            if actions:
                result = ProcessingResult(traces=traces, actions=actions)
            else:
                result = ProcessingResult(traces=traces, actions=[])
            result.latency_ms = (time.perf_counter() - started) * 1000
            if result.actions or trace.decision == Decision.IGNORE:
                return result
            # Defensive final guard for an unexpected empty non-ignore outcome.
            result.traces.append(DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="taxonomy_empty_outcome_guard",
                escalation_reason=EscalationReason.INTERNAL_ERROR,
                automated=True,
            ))
            result.actions = [OutboundAction("reply", self._escalation_reply(), True)]
            return result

        matches = self.kb.match_many(message.text) or [self.kb.match(message.text)]
        traces: list[DecisionTrace] = []
        raw_actions: list[OutboundAction] = []

        for match in matches[:5]:
            question = match.subquestion or message.text
            trace, actions = await self.decide_match(match, question, conversation_context)
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

        # First-turn invariant: the curated ignore list is contextual only.
        # If an internal matcher accidentally labels the very first message as
        # noise, never silently consume it; convert it to a human fallback.
        if is_first_message and any(t.decision == Decision.IGNORE for t in traces):
            for t in traces:
                if t.decision == Decision.IGNORE:
                    t.decision = Decision.ESCALATE
                    t.escalation_reason = EscalationReason.NO_CATEGORY
                    t.decision_reason = "first_turn_noise_escalate"
                    t.automated = True
            actions = [OutboundAction("reply", self._escalation_reply(), True)]

        # Hard application invariant: every non-ignored inbound turn must have
        # at least one user-visible action. This is a final safety net against
        # future routing branches accidentally returning an empty result.
        if not actions and not any(t.decision == Decision.IGNORE for t in traces):
            fallback_trace = DecisionTrace(
                decision=Decision.ESCALATE,
                question=message.text,
                decision_reason="empty_outcome_guard",
                escalation_reason=EscalationReason.INTERNAL_ERROR,
                automated=True,
            )
            traces.append(fallback_trace)
            actions = [OutboundAction("reply", self._escalation_reply(), True)]
            logger.error("empty_processing_result_guard | entity_id=%s", message.entity_id)

        result = ProcessingResult(traces=traces, actions=actions)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

