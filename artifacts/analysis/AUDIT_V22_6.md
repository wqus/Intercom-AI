# Terra classifier audit — v22.6

## Executive finding

The system had real classification-quality risks. The main weakness was not the embedding model itself, but decision-layer interaction:
1. semantic negative evidence could terminate an otherwise recoverable classification;
2. several legacy intents had little/no explicit intent contract;
3. external-target detection could mistake capitalized generic nouns such as `Школу` for a third-party entity;
4. the ontology was much richer than the query parser for support/event/role operations, leaving the ranker with insufficient structured evidence.

## Evidence from the frozen 321-query development set

The archived annotation file (`artifacts/analysis/terra_baseline_321_gold_annotations.csv`) is a development/error-analysis set, not a fresh holdout.

On that frozen run:
- 321 annotated queries;
- category labels: 175;
- OOS: 53;
- escalate: 23;
- multi: 11;
- operator: 59.
- overall archived system top-1 match flag: 72/321 (22.4%);
- gold-in-system-top3: 108/321 (33.6%).

Those top-1 numbers should NOT be treated as the production accuracy of the new build. They reflect the archived v22.4.1 run and a strict exact-KB-id comparison.

Among archived wrong rows, the dominant decision reasons were:
- `negative_evidence_dominates`: 54;
- `supported_ambiguity`: 74;
- `user_requested_operator`: 59 (correctly aligned with the OPERATOR gold class; it is only "wrong" under the exact-KB-id flag);
- `insufficient_domain_support`: 16;
- `unsupported_or_ambiguous`: 16;
- `wrong_entity`: 3.

The two most actionable classifier defects were therefore the semantic-negative veto and insufficient structured intent evidence.

## Changes applied

### 1. Semantic-negative rescue
Added a conservative alternative-promotion step after contrastive ranking.

It only activates when:
- the top candidate is blocked only by semantic-negative similarity;
- it has no explicit lexical/constraint contradiction;
- a lower candidate has independent evidence (direct support, lexical/fuzzy evidence or a proven contract);
- the lower candidate has adequate intent fit, real evidence, domain support and ontology alignment;
- its score is still close to the displaced candidate.

Explicit negative phrases, forbidden constraints and hard ontology mismatches remain blocking.

### 2. Structured ontology expansion
Added query/object vocabulary for:
- events;
- facilities;
- support/curator;
- feedback;
- second-person event registration;
- online mentorship;
- speaker/collaboration;
- repeated registration.

Added corresponding ontology mappings and action/state signals.

### 3. Legacy intent hardening
Added explicit contracts/signals to sparse legacy intents:
- `kb_040` online mentorship;
- `kb_041` speaker/collaboration;
- `kb_042` mentee enrollment;
- `kb_043` repeat registration;
- `kb_045` event access;
- `kb_046` second-person registration;
- `kb_022` support process;
- `kb_031` feedback.

### 4. Mentor-role disambiguation
Introduced `mentor_enrolment` query facet so:
- `Как записаться к наставнику?`
does not fall back to `personal_mentor`.

### 5. External-target false-positive fix
Generic capitalized nouns/forms such as:
`Школу`, `школу`, `Курс`, `Мероприятие`, `Семинар`
are no longer treated as third-party entities by the narrow external-target detector.

This specifically prevents false `wrong_entity` on queries like:
`как записаться в Школу по закупкам из Китая`.

## Validation

- Full test suite: **84 passed**.
- Added regression tests for:
  - semantic-negative alternative rescue;
  - explicit-negative protection;
  - mentor enrollment vs personal mentor;
  - support/event/role ontology extraction;
  - legacy role hard rules;
  - external-target false positive.

A full 321-query inference rerun was **not possible in this packaging environment** because `sentence-transformers` is not installed, so I am not claiming a measured new 321 accuracy. The archived benchmark is used only to identify and target real failure modes.

## Important next measurement

The next authoritative quality check should be a fresh holdout / production sample, not tuning on the same 321 queries. Compare:
- intent top-1 accuracy;
- top-3 recall;
- OOS precision/recall;
- wrong-template rate;
- abstention rate;
- confusion pairs (`registration`, school lifecycle, mentor roles, Telegram, payments);
- calibration of confidence/margin by intent family.

The new build deliberately avoids broad threshold relaxation; it strengthens semantic interpretation and evidence consistency first.
