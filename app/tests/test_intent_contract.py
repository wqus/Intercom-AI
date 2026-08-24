from __future__ import annotations

import json
from pathlib import Path

from app.services.knowledge_base import KnowledgeBase


KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"


def _entries():
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    return {x["id"]: x for x in data if x.get("id") != "default"}


def test_query_ontology_distinguishes_membership_from_training_price():
    membership = KnowledgeBase._query_ontology("Здравствуйте! хочу узнать стоимость вступления в клуб и условия")
    training = KnowledgeBase._query_ontology("сколько стоит обучение")

    assert "price" in membership["facet"]
    assert "membership" in membership["object"]
    assert "training" not in membership["object"]

    assert "price" in training["facet"]
    assert "training" in training["object"]
    assert "membership" not in training["object"]


def test_ontology_contract_for_known_confusing_intents():
    entries = _entries()
    assert KnowledgeBase._derive_ontology(entries["kb_012"]) == {
        "domain": "payments",
        "object": "training",
        "action": "pay",
        "state": "information",
        "facet": "price",
    }
    assert KnowledgeBase._derive_ontology(entries["kb_002"])["state"] == "failure"
    assert KnowledgeBase._derive_ontology(entries["kb_033"])["action"] == "register"
    assert KnowledgeBase._derive_ontology(entries["kb_023"])["object"] == "mentor_role"


def test_membership_query_cannot_have_strong_intent_fit_for_training_price():
    entries = _entries()
    kb = KnowledgeBase.__new__(KnowledgeBase)
    entry = entries["kb_012"]
    entry["ontology"] = KnowledgeBase._derive_ontology(entry)

    fit, reason = kb._intent_fit(
        "хочу узнать стоимость вступления в клуб",
        entry,
        fuzzy_phrase=0.70,
    )

    assert fit <= 0.30
    assert "price" in reason or "facet" in reason


def test_query_ontology_adds_action_and_state_for_membership_and_training_price():
    membership = KnowledgeBase._query_ontology("стоимость вступления в клуб")
    training = KnowledgeBase._query_ontology("сколько стоит обучение")

    assert membership["object"] == {"membership"}
    assert "join" in membership["action"]
    assert "information" in membership["state"]

    assert training["object"] == {"training"}
    assert "pay" in training["action"]
    assert "information" in training["state"]


def test_ontology_alignment_rejects_membership_as_training_price():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    entry = {
        "id": "kb_012",
        "family": "payments",
        "intent_type": "education_price",
        "intent_facets": ["price"],
        "ontology": KnowledgeBase._derive_ontology({
            "family": "payments",
            "intent_type": "education_price",
            "intent_facets": ["price"],
        }),
    }

    score, reason = kb._ontology_alignment(
        "хочу узнать стоимость вступления в клуб",
        entry,
    )

    assert score < 0.60
    assert "mismatch" in reason


def test_external_target_hint_is_conservative():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    assert kb._external_target_hint("сколько стоит обучение в Гарварде") == "гарварде"
    assert kb._external_target_hint("сколько стоит обучение в Терре") is None
    assert kb._external_target_hint("сколько стоит обучение") is None
