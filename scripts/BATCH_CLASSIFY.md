# Offline batch classification

Put one question per line into a TXT file, or use CSV with a `question` column.

```bash
PYTHONPATH=. python scripts/batch_classify.py questions.txt
```

Optional:

```bash
PYTHONPATH=. python scripts/batch_classify.py questions.csv --output-dir artifacts/batch_runs
```

Each run creates a timestamped directory containing:

- `results.csv` — spreadsheet-friendly one-row-per-input results;
- `results.jsonl` — detailed per-query results with Top-3 diagnostics;
- `summary.json` — run counts and latency summary.

The runner only instantiates `KnowledgeBase`. It does not write to the database,
send Telegram escalation notifications, call Intercom, or mutate conversation state.
