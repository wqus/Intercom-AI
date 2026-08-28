# Professional audit — Terra v22.8 final conservative router

## Executive assessment

The system is now intentionally optimized for **precision and abstention** rather than maximum coverage. The central design rule is:

> Do not answer a query merely because a KB intent is semantically nearby.

The current version is suitable as a pre-AI deterministic routing layer. It is deliberately not an LLM answerer.

## Real issues corrected in this pass

1. **Hidden Top-3 state desynchronization**
   - A disabled user-visible Top-3 could still produce `Decision.CHOICE`, making the orchestration layer capable of persisting `CHOOSING` even though no menu was sent.
   - Fixed: hidden Top-3 now becomes silent `IGNORE` and cannot create a choice state.

2. **Top-3 nearest-neighbour fallback**
   - A strong-ish single candidate could fall back to a menu built from the first three ranked retrieval results.
   - This violated the semantic requirement that Top-3 be genuine ambiguity.
   - Fixed: removed this path entirely.

3. **Weakly supported Top-3 candidates**
   - Previous filters allowed candidates to survive with modest evidence if they were semantically close.
   - Fixed: every visible candidate needs routability, no hard contradictions, meaningful real-intent evidence, and shared semantic context with the leading intent.

4. **Automatic escalation on invalid Top-3 free text**
   - Repeated non-numeric text while `CHOOSING` could escalate to a human even when the text was simply unsupported or unrelated.
   - Fixed: first invalid attempt can receive one numeric reminder; repeated non-actionable text silently closes the stale choice state instead of escalating.

5. **AI-path leakage into Top-3**
   - The LLM-mode orchestration could manufacture a user-facing Top-3 fallback when AI was disabled.
   - Fixed: only the already-safe, visible Top-3 route is permitted; otherwise the system silently abstains.

## Current decision contract

| Situation | Current behavior |
|---|---|
| Strong, proven supported intent | Deterministic KB answer/clarification |
| Genuine ambiguity between supported intents | Top-3 |
| Nearby but unsupported question | Silent OOS/IGNORE |
| No supported intent / weak evidence | Silent IGNORE |
| Unknown conversation state | Silent IGNORE |
| One-word input in ordinary idle routing | IGNORE |
| One-word input in explicit contextual choice/clarification state | Contextual resolution only when explicitly configured |
| Explicit operator request | Escalation |
| High-risk supported flow | Existing explicit escalation policy |

## Top-3 quality contract

A menu is emitted only when:
- at least two candidates survive hard contradiction/contract checks;
- both have their own intent evidence;
- both are routable to a deterministic answer/clarification;
- candidates are close enough to the leader (`<= 0.14` score gap);
- the leader is not clearly dominant (`<= 0.16` margin);
- candidates share meaningful semantic context through family and/or ontology object.

This makes Top-3 a **meaningful clarification mechanism**, not a generic uncertainty response.

## State-machine contract

The user-visible prompt and persisted state are now coupled:

`CHOICE shown -> exact candidate snapshot persisted -> numeric reply resolved from snapshot`

No user-visible menu -> no `CHOOSING` state.

## Verification

Automated suite result: **91 passed**.

Python compilation checks passed for:
- `app/application/decision_pipeline.py`
- `app/application/support_service.py`
- `app/infrastructure/ai/knowledge_base.py`

The full embedding benchmark is not claimed here because the runtime environment does not contain the optional `sentence-transformers` model dependency. The release decision is therefore based on deterministic unit/integration routing tests and source-level consistency checks, not an invented accuracy number.

## Recommended future AI layer boundary

Do not move AI into the classifier core. Keep the current deterministic system as a gate and use a future model only after:

`scope -> supported-intent checks -> deterministic routing -> controlled AI fallback`

The AI layer should never be allowed to bypass the deterministic open-set and state contracts.
