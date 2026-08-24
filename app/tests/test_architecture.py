from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_business_metrics_are_documented():
    path=ROOT.parent/"BUSINESS_VALUE.md"
    text=path.read_text(encoding="utf-8")
    for marker in ("75%", "25–30%", "1000+", "3000+", "<5 секунд"):
        assert marker in text

def test_webhook_is_thin():
    text=(ROOT/"api/webhook.py").read_text(encoding="utf-8")
    assert "KnowledgeBase" not in text
    assert "YandexLLMClient" not in text

def test_no_in_memory_webhook_dedup():
    text=(ROOT/"api/webhook.py").read_text(encoding="utf-8")
    assert "processed_events = set()" not in text

