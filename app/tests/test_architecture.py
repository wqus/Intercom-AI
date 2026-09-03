from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_project_has_production_changelog():
    project_root = ROOT.parent
    candidates = list(project_root.glob("CHANGELOG_*.md"))
    assert candidates, "production changelog must be shipped with the project"

def test_webhook_is_thin():
    text=(ROOT/"api/webhook.py").read_text(encoding="utf-8")
    assert "KnowledgeBase" not in text
    assert "YandexLLMClient" not in text

def test_no_in_memory_webhook_dedup():
    text=(ROOT/"api/webhook.py").read_text(encoding="utf-8")
    assert "processed_events = set()" not in text


