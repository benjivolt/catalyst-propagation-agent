# Verified Evidence Graph

The Verified Evidence Graph is a report-publishing trust layer for the Catalyst Propagation Agent. It turns each generated claim into a verifiable evidence object instead of treating citations as simple links.

## Pipeline

```text
source_docs
-> source_snapshots
-> source_paragraphs
-> atomic_claims
-> claim_evidence_spans
-> claim_verification_summary
-> confidence caps and human-review gate
```

## Database tables

- `source_snapshots`: stores the exact source text used at analysis time, with URL, content hash, timestamp, and archive pointer.
- `source_paragraphs`: splits snapshots into citable paragraph/span units with character offsets.
- `atomic_claims`: stores independently verifiable claims connected to the original report/effect/claim.
- `claim_evidence_spans`: stores exact quotes, paragraph IDs, retrieval score, coverage score, entailment label, contradiction score, and verifier model.
- `claim_verification_summary`: stores final status, confidence cap, required source type, publishability, and failure reason.

## Claim statuses

- `verified`: exact span directly supports the claim with a preferred source type.
- `weakly_verified`: source is relevant but support is partial or source type is not ideal.
- `unsupported`: source is related but does not prove the claim.
- `contradicted`: the span contains opposing evidence.
- `needs_human_review`: enough relevance exists to inspect manually, but the system cannot safely publish as a strong fact.

## Confidence caps

The system caps claims and effect confidence using strict rules:

- No source: max 20%.
- No exact quote: max 45%.
- Unsupported: max 35%.
- Contradicted: max 35%.
- Not enough information: max 50%.
- Partial support: max 60%.
- Regulatory claim without preferred regulatory/company/SEC source: max 60%.
- Peer relationship claim without peer evidence: max 55%.

## UI

Each report page now includes a Verified Evidence Graph panel and claim-level evidence cards showing:

- verification status,
- entailment score,
- confidence cap,
- exact quote,
- verifier reason.

## Endpoints

```text
GET  /api/reports/{report_id}/evidence-graph
POST /api/reports/{report_id}/verify-evidence-graph
```

## Test

```bash
python scripts/test_verified_evidence_graph.py
```
