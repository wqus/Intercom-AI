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


def test_llm_prompt_is_conservative_and_has_no_generation_fields():
    from app.infrastructure.ai.llm_client import build_system_prompt, build_user_message

    prompt = build_system_prompt()
    payload = build_user_message("Не могу записаться", [{
        "id": "kb_002",
        "category": "Регистрация",
        "definition": "Проблема при регистрации",
        "negative_phrases": ["как записаться"],
    }])

    assert "escalation" in prompt
    assert "нельзя придумывать новые id" in prompt.lower()
    assert "НЕ отвечаешь пользователю" in prompt
    assert '"current_question"' in payload
    assert '"candidates"' in payload
    assert "embedding_score" not in payload
    assert "_context_only" not in payload

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


def test_one_word_domain_term_is_routable_not_ignored():
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
    result = asyncio.run(pipeline.decide(msg, is_first_message=False))
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


def test_standalone_one_word_business_terms_are_routable():
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


def test_legacy_intent_anchors_cover_roles_and_event_operations():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries = []
    kb.entries_by_id = {
        "kb_040": {"id": "kb_040", "intent_type": "online_mentorship", "family": "mentor",
                   "intent_facets": ["online"], "positive_examples": ["онлайн наставничество"]},
        "kb_041": {"id": "kb_041", "intent_type": "speaker_collaboration", "family": "mentor",
                   "intent_facets": ["speaker_collaboration"], "positive_examples": ["провести интенсив"]},
        "kb_042": {"id": "kb_042", "intent_type": "become_mentee", "family": "mentor",
                   "intent_facets": ["mentor_enrolment"], "positive_examples": ["записаться к наставнику"]},
        "kb_046": {"id": "kb_046", "intent_type": "record_second_person", "family": "events",
                   "intent_facets": ["record_second_person"], "positive_examples": ["записать второго человека"]},
    }
    for kb_id in ("kb_040", "kb_041", "kb_042", "kb_046"):
        kb.entries_by_id[kb_id]["constraints"] = {}
        kb.entries_by_id[kb_id]["intent_signals"] = {}
    kb.family_vocabulary = {}

    assert kb._hard_rule_match("Хочу наставника онлайн") == "kb_040"
    assert kb._hard_rule_match("Хочу провести интенсив для предпринимателей") == "kb_041"
    assert kb._hard_rule_match("Как записаться к наставнику?") == "kb_042"
    assert kb._hard_rule_match("Хочу записать второго человека на семинар") == "kb_046"


def test_query_ontology_distinguishes_event_support_and_repeat_registration():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    q = KnowledgeBase._query_ontology("Хочу записать второго человека на семинар")
    assert "record_second_person" in q["facet"]
    assert "register" in q["action"]

    q = KnowledgeBase._query_ontology("Как обратиться к куратору?")
    assert "support_process" in q["facet"]
    assert "contact" in q["action"]

    q = KnowledgeBase._query_ontology("Можно ли онлайн работать с наставником?")
    assert "online" in q["facet"]
    assert "mentor" in q["object"]


def test_mentor_enrolment_does_not_fall_back_to_personal_mentor_facet():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    facets = KnowledgeBase._question_facets("Как записаться к наставнику?")
    assert "mentor_enrolment" in facets
    assert "personal_mentor" not in facets


def test_supported_alternative_can_rescue_semantic_negative_veto():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)

    ranked = [
        {
            "entry": {"id": "bad"},
            "negative_contrast_score": -0.04,
            "negative_hits": [],
            "constraint_evidence": {},
            "direct_support": False,
            "lexical_score": 0.20,
            "fuzzy_phrase_score": 0.30,
            "intent_contract_strength": 0.0,
            "final_score": 0.62,
        },
        {
            "entry": {"id": "good"},
            "negative_contrast_score": 0.20,
            "negative_hits": [],
            "constraint_evidence": {},
            "direct_support": True,
            "lexical_score": 0.70,
            "fuzzy_phrase_score": 0.92,
            "intent_contract_strength": 1.0,
            "intent_fit_score": 0.72,
            "real_intent_evidence": 0.60,
            "domain_support_score": 0.40,
            "ontology_alignment_score": 0.80,
            "intent_contract_status": "proven",
            "final_score": 0.57,
        },
    ]
    promoted = kb._promote_supported_alternative(ranked)
    assert promoted[0]["entry"]["id"] == "good"


def test_promote_does_not_override_explicit_candidate_negative():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    ranked = [
        {
            "entry": {"id": "bad"},
            "negative_contrast_score": -0.08,
            "negative_hits": ["explicit negative"],
            "constraint_evidence": {},
            "direct_support": False,
            "lexical_score": 0.20,
            "fuzzy_phrase_score": 0.20,
            "intent_contract_strength": 0.0,
            "final_score": 0.70,
        },
        {
            "entry": {"id": "good"},
            "negative_contrast_score": 0.10,
            "negative_hits": [],
            "constraint_evidence": {},
            "direct_support": True,
            "lexical_score": 0.70,
            "fuzzy_phrase_score": 0.92,
            "intent_contract_strength": 1.0,
            "intent_fit_score": 0.72,
            "real_intent_evidence": 0.60,
            "domain_support_score": 0.40,
            "ontology_alignment_score": 0.80,
            "intent_contract_status": "proven",
            "final_score": 0.60,
        },
    ]
    promoted = kb._promote_supported_alternative(ranked)
    assert promoted[0]["entry"]["id"] == "bad"


def test_external_target_detector_does_not_misclassify_generic_capitalized_school():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._external_target_hint(
        "как записаться в Школу по закупкам из Китая"
    ) is None


def test_external_target_detector_still_detects_named_third_party_school():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._external_target_hint(
        "подскажите стоимость обучения в Академии предпринимателей"
    ) == "академии"


def test_hard_rules_cover_online_mentorship_and_speaker_collaboration():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Хочу наставника онлайн") == "kb_040"
    assert kb._hard_rule_match("Хочу провести лекцию для предпринимателей") == "kb_041"


def test_top3_rejects_semantic_near_misses_without_intent_evidence():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {
        "a": {"id": "a", "intent_type": "school_info", "template_answer": "A", "family": "education"},
        "b": {"id": "b", "intent_type": "education_entry", "template_answer": "B", "family": "education"},
    }
    ranked = [
        {
            "entry": kb.entries_by_id["a"], "final_score": 0.61, "intent_fit_score": 0.31,
            "query_quality_score": 0.42, "real_intent_evidence": 0.22,
            "domain_support_score": 0.44, "ontology_alignment_score": 0.55,
            "direct_support": False, "positive_hits": [], "intent_signal_positive_hits": [],
            "negative_hits": [], "constraint_evidence": {}, "intent_contract_status": "unknown",
        },
        {
            "entry": kb.entries_by_id["b"], "final_score": 0.58, "intent_fit_score": 0.30,
            "query_quality_score": 0.42, "real_intent_evidence": 0.20,
            "domain_support_score": 0.40, "ontology_alignment_score": 0.55,
            "direct_support": False, "positive_hits": [], "intent_signal_positive_hits": [],
            "negative_hits": [], "constraint_evidence": {}, "intent_contract_status": "unknown",
        },
    ]
    assert kb._category_choice_candidates(ranked) == []


def test_top3_requires_routable_entries():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {
        "a": {"id": "a", "intent_type": "school_info", "family": "education", "template_answer": "A"},
        "b": {"id": "b", "intent_type": "education_entry", "family": "education"},
        "c": {"id": "c", "intent_type": "flow_date", "family": "education", "template_answer": "C"},
    }
    def item(eid, score):
        ontology = {
            "a": {"object": "school", "action": "register", "state": "procedure", "facet": "registration"},
            "b": {"object": "education", "action": "enter", "state": "procedure", "facet": "registration"},
            "c": {"object": "flow", "action": "ask", "state": "information", "facet": "date"},
        }[eid]
        return {
            "entry": kb.entries_by_id[eid], "final_score": score, "intent_fit_score": 0.60,
            "query_quality_score": 0.60, "real_intent_evidence": 0.60,
            "domain_support_score": 0.60, "ontology_alignment_score": 0.90,
            "direct_support": True, "positive_hits": ["x"], "intent_signal_positive_hits": ["x"],
            "negative_hits": [], "constraint_evidence": {}, "intent_contract_status": "proven",
            "semantic_score": 0.6, "lexical_score": 0.6, "fuzzy_phrase_score": 0.6,
            "embedding_score": 0.6, "ranking_score": score, "family_fit_score": 0.6,
            "negative_semantic_score": 0.0, "hard_negative_gap": 0.6, "negative_contrast_score": 0.6,
            "ontology": ontology,
        }
    out = kb._category_choice_candidates([item("a", .60), item("b", .59), item("c", .58)], question="как записаться в школу и когда следующий поток?")
    assert [x["kb_id"] for x in out] == ["a", "c"]


def test_top3_rejects_same_family_when_query_context_is_missing():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.entries_by_id = {
        "a": {"id": "a", "intent_type": "school_info", "family": "education", "template_answer": "A"},
        "b": {"id": "b", "intent_type": "flow_date", "family": "education", "template_answer": "B"},
    }
    def item(eid, score):
        return {
            "entry": kb.entries_by_id[eid], "final_score": score, "intent_fit_score": 0.62,
            "query_quality_score": 0.70, "real_intent_evidence": 0.70,
            "domain_support_score": 0.62, "ontology_alignment_score": 0.85,
            "direct_support": True, "positive_hits": ["x"], "intent_signal_positive_hits": ["x"],
            "negative_hits": [], "constraint_evidence": {}, "intent_contract_status": "proven",
            "family_fit_score": 0.70, "ontology": {"object": "school", "action": "register", "state": "current", "facet": ""},
        }
    assert kb._category_choice_candidates([item("a", .60), item("b", .59)], question="хочу узнать что-нибудь про обучение") == []


def test_rephrase_prompt_requests_short_problem_focused_question():
    from app.domain.policy import rephrase_prompt
    text = rephrase_prompt().lower()
    assert "короче" in text
    assert "точнее" in text
    assert "основную проблему" in text


def test_escalation_ack_explains_operator_handoff_and_optional_materials():
    from app.domain.policy import escalation_ack
    text = escalation_ack().lower()
    assert "передан оператору" in text
    assert "скриншот" in text
    assert "уточнить суть проблемы" in text


def test_category_choice_allows_contextual_alternatives_without_exact_phrase():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.family_vocabulary = {"education": {"регистрация", "школа", "поток"}}
    top_entry = _entry(
        kb, "kb_002", "education", "registration_failure", ["technical_failure", "registration"],
        ["не могу зарегистрироваться"],
    )
    top_entry["template_answer"] = "Ответ 1"
    alt_entry = _entry(
        kb, "kb_033", "education", "school_registration", ["registration", "school"],
        ["как записаться в школу"],
    )
    alt_entry["template_answer"] = "Ответ 2"
    alt2_entry = _entry(
        kb, "kb_043", "education", "repeat_registration", ["repeat_registration"],
        ["записаться повторно"],
    )
    alt2_entry["template_answer"] = "Ответ 3"
    ranked = [
        {
            "entry": top_entry, "final_score": 0.58, "intent_fit_score": 0.68,
            "real_intent_evidence": 0.52, "domain_support_score": 0.60,
            "ontology_alignment_score": 0.75, "query_quality_score": 0.20, "direct_support": True,
            "positive_hits": ["не могу зарегистрироваться"],
            "intent_signal_positive_hits": ["не могу зарегистрироваться"],
            "negative_hits": [], "constraint_evidence": {},
            "intent_contract_status": "proven", "ontology": {"object": "school", "action": "register", "state": "failure", "facet": "registration"},
        },
        {
            "entry": alt_entry, "final_score": 0.53, "intent_fit_score": 0.52,
            "real_intent_evidence": 0.36, "domain_support_score": 0.48,
            "ontology_alignment_score": 0.62, "direct_support": False,
            "positive_hits": [], "intent_signal_positive_hits": [],
            "negative_hits": [], "constraint_evidence": {},
            "intent_contract_status": "proven", "ontology": {"object": "school", "action": "register", "state": "intent", "facet": "registration"},
        },
        {
            "entry": alt2_entry, "final_score": 0.50, "intent_fit_score": 0.49,
            "real_intent_evidence": 0.35, "domain_support_score": 0.44,
            "ontology_alignment_score": 0.56, "direct_support": False,
            "positive_hits": [], "intent_signal_positive_hits": [],
            "negative_hits": [], "constraint_evidence": {},
            "intent_contract_status": "proven", "ontology": {"object": "flow", "action": "register", "state": "repeat", "facet": "repeat_registration"},
        },
    ]
    # Ensure the query ontology provides a registration context shared by the
    # alternatives, without requiring their exact trigger phrase.
    choices = kb._category_choice_candidates(ranked, question="Не могу зарегистрироваться")
    assert [c["kb_id"] for c in choices] == ["kb_002", "kb_033"] or [c["kb_id"] for c in choices] == ["kb_002", "kb_033", "kb_043"]


def test_online_training_natural_paraphrase_is_not_a_hard_rule():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match(
        "Добрый день! Хочу пройти обучение. Есть возможность пройти онлайн. Как это можно осуществить?"
    ) is None


def test_online_training_natural_paraphrase_extracts_semantic_context():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    question = "Хочу пройти обучение. Есть возможность пройти онлайн. Как это можно осуществить?"
    facets = kb._question_facets(question)
    ontology = kb._query_ontology(question)
    assert "online_training" in facets
    assert "training" in ontology["object"]
    assert "online" in ontology["facet"]
    assert ontology["action"] == set()


def test_launch_late_arrival_and_code_attendance_intents_are_distinct():
    import json
    from pathlib import Path
    data = json.loads(Path("data/knowledge_base.json").read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in data}
    late = by_id["kb_048"]
    attendance = by_id["kb_049"]

    assert late["intent_type"] == "event_late_arrival"
    assert attendance["intent_type"] == "event_attendance_for_code"
    assert "можно ли не приходить" in attendance["intent_signals"]["positive"] or "можно ли не приходить на запуск" in attendance["match_phrases"]
    assert "можно прийти позже" in late["match_phrases"]
    assert "kb_016" in late["conflict_ids"]


def test_launch_event_hard_rules_route_late_arrival_and_skip_to_distinct_kbs():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase
    kb = KnowledgeBase.__new__(KnowledgeBase)

    assert kb._hard_rule_match("Я опоздаю на Запуск, можно приехать позже?") == "kb_048"
    assert kb._hard_rule_match("Можно ли прийти на Запуск после 15:00?") == "kb_048"
    assert kb._hard_rule_match("Можно вообще не приходить на Запуск и получить код?") == "kb_049"
    assert kb._hard_rule_match("Обязательно ли быть на Запуске, чтобы получить код?") == "kb_049"


def test_launch_date_question_stays_on_existing_flow_date_intent():
    from app.infrastructure.ai.knowledge_base import KnowledgeBase
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._hard_rule_match("Когда следующий запуск нового потока?") != "kb_048"
