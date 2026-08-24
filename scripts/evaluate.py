import argparse, json
from pathlib import Path
from app.infrastructure.ai.knowledge_base import KnowledgeBase

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--kb",default="data/knowledge_base.json")
    args=ap.parse_args()
    rows=json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    kb=KnowledgeBase(path=args.kb)
    correct=0
    for row in rows:
        result=kb.match(str(row["question"]))
        predicted=result.kb_id if result.kb_id else "unknown"
        ok=predicted==row["expected"]
        correct+=ok
        if not ok:
            print(f"FAIL | expected={row['expected']} predicted={predicted} | {row['question']}")
    print(f"accuracy={correct/max(1,len(rows)):.4f} ({correct}/{len(rows)})")

if __name__=="__main__":
    main()

