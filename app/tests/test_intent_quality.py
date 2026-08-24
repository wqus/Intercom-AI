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
