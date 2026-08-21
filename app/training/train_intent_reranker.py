from __future__ import annotations

"""Fine-tune a CrossEncoder for this exact intent task.

Input JSONL format:
  {"question":"...", "intent":"kb_033"}

The script creates positive pairs (question, intent card) and hard negatives
from declared conflict_ids. A generic reranker should NOT be fine-tuned on the
KB phrases alone; real labeled user questions are preferred.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import CrossEncoder, CrossEncoderTrainer, losses
from sentence_transformers.training_args import CrossEncoderTrainingArguments


def load_kb(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in data if isinstance(row, dict) and row.get("id") != "default"}


def card(entry: dict) -> str:
    return (
        f"Семейство: {entry.get('family','')}. "
        f"Категория: {entry.get('category','')}. "
        f"Определение: {entry.get('definition', entry.get('scope', entry.get('description','')))}. "
        f"Примеры: {'; '.join(entry.get('positive_examples', entry.get('match_phrases', []))[:10])}. "
        f"Не использовать: {'; '.join(entry.get('negative_phrases', [])[:6])}."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_jsonl")
    ap.add_argument("--kb", default="knowledge_base_v2.json")
    ap.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--output", default="models/terra-intent-reranker")
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    kb = load_kb(args.kb)
    rows = []
    for raw in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        q = str(item["question"]).strip()
        intent = str(item["intent"]).strip()
        if intent not in kb:
            raise ValueError(f"Unknown intent: {intent}")

        positive = card(kb[intent])
        rows.append({"sentence1": q, "sentence2": positive, "label": 1.0})

        conflicts = kb[intent].get("conflict_ids", [])
        for conflict_id in conflicts[:4]:
            if conflict_id in kb:
                rows.append({"sentence1": q, "sentence2": card(kb[conflict_id]), "label": 0.0})

    ds = Dataset.from_list(rows).shuffle(seed=42)
    model = CrossEncoder(args.base_model, num_labels=1, max_length=512)
    loss = losses.BinaryCrossEntropyLoss(model)
    train_args = CrossEncoderTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="epoch",
        load_best_model_at_end=False,
        fp16=False,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(args.output)
    print(f"saved: {args.output}; pairs: {len(ds)}")


if __name__ == "__main__":
    main()
