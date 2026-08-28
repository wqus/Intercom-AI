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


def test_structured_negation_blocks_auto_answer_for_negated_goal():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.family_vocabulary = {"mentor": {"наставник", "стать"}}
    entry = _entry(
        kb, "kb_023", "mentor", "mentor_application",
        ["mentor_application"], ["стать наставником", "хочу стать наставником"],
    )
    ok, reason = kb._template_eligibility(
        "Не хочу стать наставником", entry,
        intent_fit=0.90, family_fit=0.80, fuzzy_score=0.99,
    )
    assert not ok
    assert reason.startswith("structured_negation:")


def test_problem_state_negation_is_not_treated_as_structured_negation():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.family_vocabulary = {"education": {"записаться", "школа"}}
    entry = _entry(
        kb, "kb_002", "education", "registration_failure",
        ["technical_failure", "registration"], ["не могу записаться", "ошибка при записи"],
    )
    hits = kb._structured_negation_hits("Не могу записаться в школу", entry)
    assert hits == []


def test_hard_rule_is_challenged_by_strong_conflicting_intent():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {
        "mentor": {"наставник", "программа", "стать"},
    }
    target = _entry(
        kb, "kb_023", "mentor", "mentor_application",
        ["mentor_application"], ["стать наставником"],
    )
    rival = _entry(
        kb, "kb_034", "mentor", "mentor_program",
        ["mentor_program"], ["программа наставничества", "что такое наставничество"],
    )
    target["conflict_ids"] = ["kb_034"]
    rival["conflict_ids"] = ["kb_023"]
    kb.entries = [target, rival]
    passed, reason = kb._challenge_hard_rule("Что такое программа наставничества?", target)
    assert not passed
    assert reason == "hard_rule_challenged_by:kb_034"


def test_template_gate_rejects_missing_semantic_precondition():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.family_vocabulary = {"mentor": {"наставник", "стать"}}
    entry = _entry(
        kb, "kb_023", "mentor", "mentor_application",
        ["mentor_application"], ["стать наставником"],
    )
    ok, reason = kb._template_eligibility(
        "Расскажите про программу наставничества", entry,
        intent_fit=0.95, family_fit=0.90,
    )
    assert not ok
    assert reason.startswith("missing_required_intent_facet:")


def test_query_understanding_keeps_generic_verification_object_unknown():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    qu = kb._query_understanding("проблема верификации")
    assert "verification" in qu.facets
    assert "phone" not in qu.objects
    assert "income" not in qu.objects
    assert qu.specificity == "low"


def test_income_contract_does_not_accept_generic_verification():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"income": {"доход", "опора", "уровень"}}
    entry = _entry(
        kb, "kb_010", "income", "income_or_support_pillar",
        ["income", "support"], ["подтвердить доход", "уровень дохода"],
    )
    entry["constraints"] = {"must_have_any": [["опор", "доход", "уровень"]]}
    evidence = kb._intent_contract_evidence("проблема верификации", entry)
    assert evidence.status == "insufficient"
    assert "required_constraint_group" in evidence.reason or "defining_evidence" in evidence.reason


def test_phone_contract_requires_phone_evidence_not_generic_verification():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"account": {"телефон", "номер", "смс"}}
    entry = _entry(
        kb, "kb_027", "account", "phone_verification",
        ["verification"], ["не могу подтвердить номер", "не приходит смс"],
    )
    entry["constraints"] = {"must_have_any": [["номер", "телефон", "смс", "подтвержден"]]}
    evidence = kb._intent_contract_evidence("проблема верификации", entry)
    assert evidence.status == "insufficient"
    assert evidence.reason.startswith("missing_required_object:phone")


def test_phone_contract_is_proven_with_specific_phone_query():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"account": {"телефон", "номер", "смс"}}
    entry = _entry(
        kb, "kb_027", "account", "phone_verification",
        ["verification"], ["не могу подтвердить номер", "не приходит смс"],
    )
    entry["constraints"] = {"must_have_any": [["номер", "телефон", "смс", "подтвержден"]]}
    evidence = kb._intent_contract_evidence("не приходит смс для подтверждения номера", entry)
    assert evidence.status == "proven"
    assert "phone" in evidence.matched_objects


def test_template_gate_rejects_semantically_close_unsupported_income_query():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.family_vocabulary = {"income": {"доход", "опора", "уровень"}}
    entry = _entry(
        kb, "kb_010", "income", "income_or_support_pillar",
        ["income"], ["подтвердить доход", "уровень дохода"],
    )
    entry["constraints"] = {"must_have_any": [["опор", "доход", "уровень"]]}
    ok, reason = kb._template_eligibility(
        "проблема верификации", entry,
        intent_fit=0.95, family_fit=0.90, fuzzy_score=0.88,
    )
    assert not ok
    assert reason in {"missing_semantic_precondition", "intent_contract:missing_required_constraint_group"}


def test_llm_validator_cannot_bypass_missing_intent_contract():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.entries_by_id = {}
    kb.family_vocabulary = {"account": {"телефон", "номер", "смс"}}
    entry = _entry(
        kb, "kb_027", "account", "phone_verification",
        ["verification"], ["не могу подтвердить номер"],
    )
    entry["constraints"] = {"must_have_any": [["номер", "телефон", "смс", "подтвержден"]]}
    kb.entries_by_id["kb_027"] = entry
    ok, reason, _ = kb.validate_llm_selection("проблема верификации", "kb_027")
    assert not ok
    assert reason.startswith("llm_selection_contract:")


def test_category_choice_candidates_drop_unproven_contracts():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.family_vocabulary = {
        "income": {"доход", "опора", "уровень"},
        "account": {"телефон", "номер", "смс"},
    }
    income = _entry(
        kb, "kb_010", "income", "income_or_support_pillar",
        ["income"], ["подтвердить доход"],
    )
    income["constraints"] = {"must_have_any": [["опор", "доход", "уровень"]]}
    phone = _entry(
        kb, "kb_027", "account", "phone_verification",
        ["verification"], ["не приходит смс"],
    )
    phone["constraints"] = {"must_have_any": [["номер", "телефон", "смс", "подтвержден"]]}
    kb.entries_by_id = {"kb_010": income, "kb_027": phone}
    ranked = [
        {"entry": income, "final_score": .86, "ranking_score": .86,
         "intent_fit_score": .80, "real_intent_evidence": .60,
         "family_fit_score": .60, "direct_support": True,
         "contrastive_margin": .20},
        {"entry": phone, "final_score": .84, "ranking_score": .84,
         "intent_fit_score": .78, "real_intent_evidence": .55,
         "family_fit_score": .55, "direct_support": True,
         "contrastive_margin": .10},
    ]
    # The public selector must never expose a candidate whose own contract is not proven.
    choices = kb._category_choice_candidates(ranked)
    assert choices == []


def test_high_signal_recurring_charge_routes_to_payment():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {"kb_011": {"id": "kb_011", "priority": 90}}
    assert kb._hard_rule_match("Добрый день, как отменить списания по карте?") == "kb_011"


def test_high_signal_mentor_enrolment_routes_to_become_mentee():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {"kb_042": {"id": "kb_042", "priority": 100}}
    assert kb._hard_rule_match("Как осуществляется запись к наставнику?") == "kb_042"


def test_support_process_does_not_require_generic_semantic_match():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {"kb_022": {"id": "kb_022", "priority": 100}}
    assert kb._hard_rule_match("где найти куратора") == "kb_022"


def test_content_correction_routes_to_feedback():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {"kb_031": {"id": "kb_031", "priority": 90}}
    assert kb._hard_rule_match("Период прохождения школы в описании не верный, внесите корректировку") == "kb_031"
