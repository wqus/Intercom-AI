# Terra classifier audit — v22.7

## Scope of this pass

The requested focus was not generic accuracy tuning, but three production failure modes:
1. questions that are semantically near the KB but are not actually supported;
2. Top-3 menus followed by a numeric reply that occasionally escalates;
3. one-word input and states/intents that are not explicitly implemented.

## Real risks found

### A. Open-set false positives
The classifier intentionally had a broad middle path where semantic similarity could lead to Top-3 even without strong intent evidence. In `_category_choice_candidates()`, a permissive fallback could select the nearest ranked entries after strict filters failed.

That is unsafe for the requested use case: a near-topic question can be close in embedding space while lacking the action/state/object that defines a supported KB intent.

**Fix:** Top-3 now requires either explicit intent evidence (`direct_support`, positive phrase evidence or positive intent signal) or unusually strong structured evidence across intent fit, real intent evidence, domain support and ontology alignment. The fallback that took arbitrary nearest candidates was removed.

### B. Top-3 -> numeric reply was state-fragile
There was already a `CHOOSING` state, but its reliability depended on the exact branch that produced the menu. Some paths could produce a visible Top-3 from broad ranked candidates even when the downstream state was not guaranteed to be resolvable. Selection also depended on looking up the KB entry again.

**Fix:**
- only routable entries may appear in Top-3;
- the exact candidate IDs shown are persisted;
- answer snapshots are persisted with them;
- `1/2/3` is consumed before generic classification;
- answer resolution first uses the persisted snapshot.

This makes the UI promise and state-machine promise identical.

### C. One-word messages were treated inconsistently
Query quality already marked one-word input as `single_word`, but the public non-query gate still allowed one-word domain nouns such as `Оплата` or `Телеграм` into classification.

**Fix:** standalone one-word messages are now ignored by default. Explicit operator requests are checked first, and contextual short replies are handled by state-specific logic.

### D. Unsupported states / missing routable actions
The application previously had a finite state enum, but no explicit guard for a legacy/future state value. Also, the LLM-disabled fallback could emit a rephrase when no safe routable choice existed.

**Fix:** unknown state values are ignored; unsupported `NO_MATCH` and no-routable LLM outcomes are silent.

## New routing contract

The effective policy is now:

`supported + decisive` → answer/clarify

`supported + ambiguous + 2+ routable intents` → Top-3 + CHOOSING state

`known but not safely routable` → silent abstention

`nearby / semantically similar but unsupported` → silent abstention

`one word` → silent abstention unless an explicit conversation state defines that short reply

`operator / explicit high-risk` → escalation path

## Validation

- Full suite: **90 passed**.
- Added tests for:
  - standalone one-word suppression;
  - operator exception;
  - silent OOS;
  - Top-3 rejection of semantic near-misses;
  - Top-3 routability;
  - contextual one-word clarification;
  - persisted Top-3 answer snapshots;
  - unsupported future state handling.

No claim is made that measured production accuracy increased without a fresh embedding benchmark. The structural changes target precision of the abstention boundary and reliability of stateful routing.
