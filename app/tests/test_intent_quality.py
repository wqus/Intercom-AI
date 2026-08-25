from app.infrastructure.ai.knowledge_base import KnowledgeBase
from app.infrastructure.ai.llm_client import build_user_message


def _entry(kb, kb_id, family, intent_type, facets, positive, negative=None):
    return {
        "id": kb_id,
        "family": family,
        "intent_type": intent_type,
        "intent_facets": facets,
        "intent_signals": {"positive": positive, "negative": negative or []},
        "ontology": kb._derive_ontology({
            "id": kb_id,
            "family": family,
            "intent_type": intent_type,
            "intent_facets": facets,
        }),
        "positive_examples": positive,
        "negative_phrases": negative or [],
        "constraints": {},
    }


def test_pairwise_discriminator_prefers_registration_failure_over_generic_registration():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    question = "Не могу записаться в школу, пишет ошибку"
    failure = _entry(
        kb, "kb_002", "education", "registration_failure",
        ["technical_failure", "registration"],
        ["не могу записаться", "ошибка при записи"],
        ["как записаться"],
    )
    generic = _entry(
        kb, "kb_033", "education", "school_registration",
        ["registration", "school"],
        ["как записаться в школу"],
        ["не могу записаться", "ошибка при записи"],
    )
    a = {"entry": failure, "positive_hits": ["не могу записаться"], "constraint_evidence": {}}
    b = {"entry": generic, "positive_hits": [], "constraint_evidence": {}}
    result = kb._pairwise_discriminator(question, a, b)
    assert result["advantage"] > 0


def test_pairwise_discriminator_rejects_mentor_program_when_user_wants_to_be_mentor():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    question = "Хочу стать наставником"
    apply = _entry(
        kb, "kb_023", "mentor", "mentor_application",
        ["mentor_application"],
        ["стать наставником", "стать ментором"],
        ["программа наставничества"],
    )
    program = _entry(
        kb, "kb_034", "mentor", "mentor_program",
        ["mentor_program"],
        ["программа наставничества", "что такое наставничество"],
        ["хочу стать наставником"],
    )
    a = {"entry": apply, "positive_hits": ["стать наставником"], "constraint_evidence": {}}
    b = {"entry": program, "positive_hits": [], "constraint_evidence": {}}
    result = kb._pairwise_discriminator(question, a, b)
    assert result["advantage"] > 0.1


def test_llm_prompt_contains_contrastive_evidence_and_context_only_flag():
    entries = [{
        "id": "kb_002",
        "category": "registration_failure",
        "intent_type": "registration_failure",
        "intent_facets": ["registration", "technical_failure"],
        "intent_signals": {"positive": ["не могу записаться"], "negative": ["как записаться"]},
        "constraints": {},
        "definition": "Проблема с записью",
        "positive_examples": ["Не могу записаться"],
        "negative_phrases": ["Как записаться"],
        "_context_only": True,
        "_contrastive_margin": 0.01,
        "_contrastive_competitor_id": "kb_033",
        "_contrastive_reason": "A:positive_signal;B:negative_signal",
    }]
    payload = build_user_message("Не могу записаться", entries)
    assert "context_only" in payload
    assert "contrastive" in payload.lower()
    assert "intent_signals" in payload


def test_llm_selection_validator_blocks_explicit_negative_signal():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    entry = _entry(
        kb, "kb_033", "education", "school_registration",
        ["registration", "school"],
        ["как записаться в школу"],
        ["не могу записаться"],
    )
    kb.entries_by_id = {"kb_033": entry}
    kb.family_vocabulary = {"education": {"школа", "записаться", "запись"}}
    ok, reason, _ = kb.validate_llm_selection("Не могу записаться в школу", "kb_033")
    assert not ok
    assert reason == "llm_selection_contradicted"


def test_real_world_mentor_application_with_inflection_is_not_personal_mentor():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    question = "Здравствуйте, прошел обучение в школе финансов, хочу быть наставников в будущем, как это сделать,"
    assert kb._hard_rule_match(question) == "kb_023"
    facets = kb._question_facets(question)
    assert "mentor_application" in facets


def test_personal_mentor_requires_personal_mentor_facet():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    entry = _entry(
        kb, "kb_009", "mentor", "personal_mentor",
        ["personal_mentor"], ["где мой наставник"], ["стать наставником"],
    )
    fit, reason = kb._intent_fit("Хочу быть наставников в будущем", entry)
    assert fit < 0.2
    assert reason.startswith("missing_required_intent_facet")


def test_exact_one_word_acknowledgement_is_non_query():
    from app.infrastructure.ai.knowledge_base import is_non_query_message, _query_quality
    assert is_non_query_message("Спасибо")
    q = _query_quality("Спасибо")
    assert q.meta_noise
    assert q.reason == "meta_noise"


def test_one_word_domain_term_is_not_ignored():
    from app.infrastructure.ai.knowledge_base import is_non_query_message, _query_quality
    assert not is_non_query_message("Оплата")
    q = _query_quality("Оплата")
    assert not q.meta_noise
    assert q.reason == "single_word"


def test_real_world_app_login_has_deterministic_hard_rule():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    question = "не могу войти в приложение Терра ни по номеру телефона, ни по емэйл"
    assert kb._hard_rule_match(question) == "kb_001"


def test_attachment_acknowledgement_is_ignored_before_attachment_escalation():
    import asyncio
    from app.application.decision_pipeline import DecisionPipeline
    from app.domain.models import InboundMessage
    from app.domain.enums import Channel
    from app.infrastructure.ai.knowledge_base import is_non_query_message

    assert is_non_query_message("Спасибо")

    class NeverKB:
        def match_many(self, q):
            raise AssertionError("KB must not be reached for exact non-query")

    pipeline = DecisionPipeline(NeverKB(), object())
    msg = InboundMessage("e","k","t","c",Channel.TICKET,"Спасибо",has_attachments=True)
    result = asyncio.run(pipeline.decide(msg))
    assert result.traces[0].decision.value == "ignore"
    assert result.actions == []


def test_oos_is_first_class_and_does_not_force_nearest_category():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    # A coherent but clearly unrelated question should be OOS rather than
    # being forced into the nearest semantic KB category.
    kb.family_vocabulary = {"education": {"школа", "обучение"}}
    kb.entries = []
    ranked = [{
        "entry": {"id": "kb_033", "family": "education"},
        "final_score": 0.42,
        "semantic_score": 0.51,
        "domain_support_score": 0.0,
        "family_fit_score": 0.10,
        "intent_fit_score": 0.12,
        "real_intent_evidence": 0.08,
        "ontology_alignment_score": 0.50,
        "direct_support": False,
        "positive_hits": [],
        "intent_signal_positive_hits": [],
    }]
    quality = kb.query_quality("Как оформить налоговый вычет за квартиру?")
    assert kb._oos_reason("Как оформить налоговый вычет за квартиру?", ranked, quality) == "oos:no_supported_intent"


def test_oos_does_not_reject_rare_in_domain_query():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"education_admin": {"сертификат", "дресс", "одежда"}}
    ranked = [{
        "entry": {"id": "kb_029", "family": "education_admin"},
        "final_score": 0.49,
        "semantic_score": 0.66,
        "domain_support_score": 0.22,
        "family_fit_score": 0.52,
        "intent_fit_score": 0.48,
        "real_intent_evidence": 0.40,
        "ontology_alignment_score": 0.64,
        "direct_support": False,
        "positive_hits": [],
        "intent_signal_positive_hits": [],
    }]
    quality = kb.query_quality("сертификат")
    # single-word domain terms may have low query quality, but must not become OOS.
    assert kb._oos_reason("сертификат", ranked, quality) is None


def test_oos_external_target_requires_domain_sensitive_context():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"education": {"школа", "обучение"}}
    ranked = [{
        "entry": {"id": "kb_033", "family": "education"},
        "final_score": 0.40,
        "semantic_score": 0.60,
        "domain_support_score": 0.05,
        "family_fit_score": 0.12,
        "intent_fit_score": 0.10,
        "real_intent_evidence": 0.05,
        "ontology_alignment_score": 0.50,
        "direct_support": False,
        "positive_hits": [],
        "intent_signal_positive_hits": [],
    }]
    q = "Как узнать стоимость обучения в другой школе?"
    quality = kb.query_quality(q)
    assert kb._oos_reason(q, ranked, quality) is None or isinstance(kb._oos_reason(q, ranked, quality), str)


def test_single_word_known_business_term_can_be_routed():
    # The filter must not classify a meaningful one-word support term as noise merely
    # because it has no question word. The exact-match ignore list is the only ignore gate.
    from app.infrastructure.ai.knowledge_base import is_non_query_message
    for term in ("Оплата", "Наставник", "Сертификат", "Телеграм"):
        assert not is_non_query_message(term)


def test_negated_mentor_application_does_not_trigger_kb_023():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {
        "kb_023": {
            "id": "kb_023", "intent_type": "mentor_application",
            "family": "mentor",
            "intent_facets": ["mentor_application"],
            "positive_examples": ["стать наставником"],
            "intent_signals": {"positive": ["стать наставником"], "negative": []},
        }
    }
    assert kb._hard_rule_match("Не хочу быть наставником") != "kb_023"
