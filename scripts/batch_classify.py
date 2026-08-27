"""Offline batch runner for the deterministic classification pipeline.

Input formats:
  * TXT: one question per non-empty line
  * CSV: reads a column named ``question`` (or the first column)
  * JSON/JSONL: accepts strings, or objects containing ``question``

The runner uses only KnowledgeBase. It does not create DB records, send
Telegram escalation notifications, call Intercom, or persist conversation
state. Outputs are written to ``artifacts/batch_runs/<run_id>/`` by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.infrastructure.ai.knowledge_base import KnowledgeBase


def load_questions(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        key = "question" if "question" in rows[0] else next(iter(rows[0]))
        return [str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()]
    if suffix == ".jsonl":
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            result.append(str(value.get("question", "")) if isinstance(value, dict) else str(value))
        return [x.strip() for x in result if x.strip()]
    if suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("questions", values.get("data", []))
        result = []
        for value in values:
            result.append(str(value.get("question", "")) if isinstance(value, dict) else str(value))
        return [x.strip() for x in result if x.strip()]
    raise ValueError(f"Unsupported input format: {suffix}. Use .txt, .csv, .jsonl or .json")


def flat_result(result: Any) -> dict[str, Any]:
    candidates = result.top_candidates or result.relevant_entries or []
    top3 = []
    for item in candidates[:3]:
        top3.append({
            "kb_id": item.get("kb_id") or item.get("entry", {}).get("id"),
            "category": item.get("category") or item.get("entry", {}).get("category"),
            "final_score": item.get("final_score", item.get("score")),
        })
    return {
        "mode": result.mode.value,
        "kb_id": result.kb_id,
        "category": result.category,
        "score": result.score,
        "second_score": result.second_score,
        "margin": result.margin,
        "embedding_score": result.embedding_score,
        "semantic_score": result.semantic_score,
        "lexical_score": result.lexical_score,
        "fuzzy_score": result.fuzzy_score,
        "evidence_score": result.evidence_score,
        "real_intent_evidence": result.real_intent_evidence,
        "intent_fit_score": result.intent_fit_score,
        "intent_fit_reason": result.intent_fit_reason,
        "family_fit_score": result.family_fit_score,
        "query_quality_score": result.query_quality_score,
        "domain_support_score": result.domain_support_score,
        "ontology_alignment_score": result.ontology_alignment_score,
        "negative_contrast_score": result.negative_contrast_score,
        "contrastive_margin": result.contrastive_margin,
        "contrastive_competitor_id": result.contrastive_competitor_id,
        "decision_reason": result.decision_reason,
        "top3": top3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Terra classification offline over many questions")
    parser.add_argument("input", type=Path)
    parser.add_argument("--kb", default="data/knowledge_base.json")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/batch_runs"))
    args = parser.parse_args()

    questions = load_questions(args.input)
    if not questions:
        print("No questions found", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / run_id
    out.mkdir(parents=True, exist_ok=False)

    kb = KnowledgeBase(path=args.kb)
    csv_path = out / "results.csv"
    jsonl_path = out / "results.jsonl"
    latencies: list[float] = []
    mode_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_fh, jsonl_path.open("w", encoding="utf-8") as jsonl_fh:
        fieldnames = [
            "index", "query", "subquestion_count", "subquestions", "modes", "kb_ids",
            "categories", "scores", "margins", "decision_reasons", "top3", "latency_ms",
        ]
        writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        writer.writeheader()

        for index, question in enumerate(questions, start=1):
            started = time.perf_counter()
            results = kb.match_many(question)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            for result in results:
                mode_counts[result.mode.value] = mode_counts.get(result.mode.value, 0) + 1

            details = [flat_result(result) for result in results]
            row = {
                "index": index,
                "query": question,
                "subquestion_count": len(results),
                "subquestions": " || ".join((r.subquestion or question) for r in results),
                "modes": " | ".join(d["mode"] for d in details),
                "kb_ids": " | ".join(str(d["kb_id"] or "") for d in details),
                "categories": " | ".join(str(d["category"] or "") for d in details),
                "scores": " | ".join(f'{float(d["score"]):.6f}' for d in details),
                "margins": " | ".join(f'{float(d["margin"]):.6f}' for d in details),
                "decision_reasons": " | ".join(str(d["decision_reason"]) for d in details),
                "top3": json.dumps([d["top3"] for d in details], ensure_ascii=False),
                "latency_ms": round(elapsed_ms, 3),
            }
            writer.writerow(row)
            payload = {"index": index, "query": question, "latency_ms": round(elapsed_ms, 3), "results": details}
            jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows.append(payload)

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted) if latencies_sorted else 0.0
    p95_index = min(len(latencies_sorted) - 1, max(0, int(round(0.95 * len(latencies_sorted))) - 1)) if latencies_sorted else 0
    summary = {
        "run_id": run_id,
        "input": str(args.input),
        "kb": str(args.kb),
        "questions": len(questions),
        "results": sum(len(row["results"]) for row in rows),
        "mode_counts": mode_counts,
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 3),
            "p50": round(p50, 3),
            "p95": round(latencies_sorted[p95_index], 3) if latencies_sorted else 0.0,
            "max": round(max(latencies), 3),
        },
        "artifacts": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "summary": str(out / "summary.json"),
        },
        "offline_isolation": {
            "database": False,
            "intercom": False,
            "telegram": False,
            "conversation_state": False,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run={run_id}")
    print(f"questions={len(questions)}")
    print(f"output={out}")
    print(f"csv={csv_path}")
    print(f"jsonl={jsonl_path}")
    print(f"summary={out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
