# Evaluation tools

`evaluate.py` evaluates routing against a JSON benchmark with fields:

```json
{"question": "...", "expected": "kb_016"}
```

Run from the project root:

```bash
PYTHONPATH=. python scripts/evaluate.py path/to/benchmark.json
```

The evaluator intentionally reports exact category accuracy only. Route/action/OOS evaluation should be added once a labeled benchmark exists.
