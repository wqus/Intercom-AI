from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from app.services.knowledge_base import KnowledgeBase, MatchMode


def load_cases(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark must be a JSON list")
    for row in data:
        if not isinstance(row, dict) or not row.get("question") or not row.get("expected"):
            continue
        yield str(row["question"]), str(row["expected"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--kb", default="data/knowledge_base.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reranker", action="store_true")
    args = ap.parse_args()

    kb = KnowledgeBase(path=args.kb, device=args.device, use_reranker=args.reranker)
    rows = list(load_cases(args.benchmark))

    y_true = []
    y_pred = []
    modes = Counter()
    errors = []

    for question, expected in rows:
        result = kb.match(question)
        predicted = result.kb_id if result.mode in (MatchMode.TEMPLATE, MatchMode.LLM, MatchMode.CLARIFICATION) else "no_match"
        y_true.append(expected)
        y_pred.append(predicted)
        modes[result.mode.value] += 1
        if predicted != expected:
            errors.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "mode": result.mode.value,
                "score": result.score,
                "second_score": result.second_score,
                "margin": result.margin,
            })

    labels = sorted(set(y_true) | set(y_pred))
    print("cases:", len(rows))
    print("accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("macro_f1:", round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4))
    print("modes:", dict(modes))
    print("\nclassification report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("confusion matrix labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nerrors:", len(errors))
    for e in errors[:50]:
        print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
