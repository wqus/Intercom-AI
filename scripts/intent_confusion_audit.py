"""Static audit of the KB confusion graph.

This does not claim model accuracy. It identifies where two intents share vocabulary
and whether the KB already contains explicit positive/negative discriminators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from rapidfuzz import fuzz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="data/knowledge_base.json")
    ap.add_argument("--out", default="CONFUSION_AUDIT.md")
    args = ap.parse_args()

    rows = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    entries = [x for x in rows if isinstance(x, dict) and x.get("id") != "default"]
    by_id = {str(x["id"]): x for x in entries}

    seen: set[tuple[str, str]] = set()
    lines = ["# Intent Confusion Audit", "", "Static analysis of declared conflict pairs.", ""]
    high_overlap = 0

    for entry in entries:
        for raw_id in entry.get("conflict_ids", []) or []:
            other_id = str(raw_id)
            if other_id not in by_id:
                continue
            pair = tuple(sorted((str(entry["id"]), other_id)))
            if pair in seen:
                continue
            seen.add(pair)
            other = by_id[other_id]

            a_pos = [str(x) for x in (entry.get("intent_signals") or {}).get("positive", [])]
            b_pos = [str(x) for x in (other.get("intent_signals") or {}).get("positive", [])]
            overlaps: list[tuple[float, str, str]] = []
            for a in a_pos:
                for b in b_pos:
                    score = fuzz.token_set_ratio(a, b)
                    if score >= 75:
                        overlaps.append((score, a, b))
            overlaps.sort(reverse=True)
            overlaps = overlaps[:3]
            if overlaps:
                high_overlap += 1

            a_neg = len((entry.get("intent_signals") or {}).get("negative", []))
            b_neg = len((other.get("intent_signals") or {}).get("negative", []))
            a_facets = set(entry.get("intent_facets", []) or [])
            b_facets = set(other.get("intent_facets", []) or [])
            facet_diff = sorted(a_facets ^ b_facets)

            lines.append(f"## {entry['id']} ↔ {other['id']}")
            lines.append(
                f"**{entry.get('intent_type')}** vs **{other.get('intent_type')}**; "
                f"families: `{entry.get('family')}` / `{other.get('family')}`."
            )
            lines.append(
                f"- Positive/negative signals: `{len(a_pos)}/{a_neg}` vs `{len(b_pos)}/{b_neg}`"
            )
            lines.append(f"- Facet difference: `{', '.join(facet_diff) or 'none'}`")
            if overlaps:
                lines.append("- Shared semantic/lexical signals:")
                for score, a, b in overlaps:
                    lines.append(f"  - {score:.0f}: `{a}` ↔ `{b}`")
            else:
                lines.append("- Shared semantic/lexical signals: none detected")
            lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        f"- Declared conflict pairs: **{len(seen)}**",
        f"- Pairs with strong lexical overlap (≥75): **{high_overlap}**",
        "- The production router should treat these pairs contrastively: high score for A is not sufficient unless A also beats B on explicit intent evidence.",
        "- Real quality measurement should be performed against historical, human-labeled Intercom conversations; KB examples alone are not a production accuracy benchmark.",
        "",
    ])
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} | conflict_pairs={len(seen)} | high_overlap={high_overlap}")


if __name__ == "__main__":
    main()
