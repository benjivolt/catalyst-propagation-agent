# Claim verification

After a report is saved, `app/services/verified_evidence_graph.py` checks every cited claim against the stored text of its sources and caps confidence where the support is thin. It runs locally with no model calls.

## Steps

1. Snapshot each source document attached to the catalyst (title, publisher, excerpt and any stored page text) into `source_snapshots`, with a content hash. Pages are not re-fetched here.
2. Split each snapshot into paragraphs with character offsets (`source_paragraphs`).
3. Store each report claim as an `atomic_claims` row and classify it: regulatory, trial result, deal, drug or target, peer relationship, propagation inference, or background.
4. Rank candidate paragraphs for the claim and score the best ones (`claim_evidence_spans`): token coverage of the claim, entity match, source quality, and a simple contradiction check for opposite-direction wording. These are lexical scores, not a trained entailment model.
5. Write one verdict per claim to `claim_verification_summary` and lower the parent effect's confidence to the lowest cap among its claims (never below 0.25).

## Verdicts and caps

| Verdict | Meaning | Cap |
| --- | --- | --- |
| verified | Passage covers the claim and the source type suits the claim | 0.88 |
| weakly verified | Partial coverage | 0.60 |
| needs human review | Related passage, not enough to decide | 0.50 |
| unsupported | Passage does not back the claim | 0.35 |
| contradicted | Passage points the other way | 0.35 |

Further limits apply on top of these: 0.20 when the claim has no source, 0.45 when no passage was quoted, 0.72 when the source is neither verified nor official, 0.55 for a peer-relationship claim not backed by a peer-relationship source, and a per-claim-type ceiling when the source type is not one the claim type prefers (for example 0.60 for a deal claim without an SEC filing or company release).

The report gets a grade from A to D based on how many claims are verified. Its overall confidence is capped at 0.92, 0.80, 0.65 or 0.48 for A to D, and any grade below A puts the report back into the review queue.

## API

```text
GET  /api/reports/{report_id}/evidence-graph
POST /api/reports/{report_id}/verify-evidence-graph
```

Tests: `tests/test_verified_evidence_graph.py`.
