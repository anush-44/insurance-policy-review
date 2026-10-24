# PolicyLens Resume Metrics

## Scope

These are **offline prototype measurements** using the repository's generated
Care Advantage sample policy (`data/sample_policy.pdf`) and its matching
hand-labeled Q&A set. They are not measurements across the claimed 15-policy
corpus. The other policy PDFs were outside the workspace and were not included.
No policy text was sent to Gemini or another external model during these checks.

## Retrieval: Hit@5

Hit@5 is counted as a hit when at least one of the top five retrieved chunks
has the expected clause ID from the labeled question.

| Retrieval method | Hits | Questions | Hit@5 |
|---|---:|---:|---:|
| BM25-only | 40 | 40 | 100% |
| Dense retrieval + BM25 fused with reciprocal-rank fusion (RRF) | 40 | 40 | 100% |
| Hybrid change over BM25 | — | 40 | 0 percentage points |

The evaluation used the 40 answerable labeled questions for `care-adv-sample`.
The 10 abstention questions were excluded because they do not have a gold clause
ID. This test set is small and limited to one generated sample policy. The
result does **not** establish general retrieval performance, nor does it show
an improvement from hybrid retrieval on this test.

**Cross-encoder reranking was not part of this measurement.** The repository
contains a reranker model setting, but no executed cross-encoder reranking step
was found in the retrieval path.

## Summary extraction: Offline fallback

| Policy | Correct labeled fields | Labeled fields | Match rate |
|---|---:|---:|---:|
| `care-adv-sample` | 4 | 13 | 30.8% |

This score compares the deterministic offline summary fallback with the one
matching hand-labeled record. A field counted as correct only when its expected
value appeared in the corresponding summary output; missing summary values
counted as incorrect. It does **not** measure live Gemini extraction accuracy
and should not be presented as accuracy across multiple real policies.

## Metrics not established

- **15-policy corpus size:** not independently verified in this workspace.
- **RAGAS faithfulness and answer relevance:** not measured; this offline run
  did not use an evaluator model.
- **Live-model extraction accuracy:** not measured; no external LLM was used.
- **Cross-encoder reranking gain:** not measured; cross-encoder reranking was
  not implemented in the tested retrieval path.
- **Docker deployment:** Docker configuration exists, but this report does not
  establish that the application was deployed.
- **Langfuse tracing:** configuration exists, but this report does not
  establish active instrumentation or collected traces.

## Resume-safe wording

> Evaluated retrieval on 40 labeled questions for a generated sample policy:
> BM25 and dense + BM25/RRF both achieved 100% Hit@5, with no measured gain on
> this test set. Cross-encoder reranking and RAGAS evaluation were not validated.

Avoid claiming 15-policy performance, live-model field accuracy, RAGAS scores,
cross-encoder gains, deployment, or active Langfuse tracing without additional
implementation and reproducible evidence.
