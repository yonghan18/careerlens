# Retrieval index consistency

Status: specification of existing retrieval behavior, covered by offline
regression tests.

## Problem and goal

Document edits can change chunk IDs while an older Qdrant index still contains
the previous IDs. CareerLens previously stopped when retrieval tried to resolve
such an ID against the current chunk catalog. A stale persisted entry must neither
crash a query nor appear as evidence about the candidate.

## Scope

Existing implementation: `CorpusStore.search()` and
`CorpusStore._prune_dense_index()` in `app/retrieval.py`.
Add deterministic tests for those boundaries; change runtime code only if the
tests expose a failure. No paid embeddings or generation are required.

## Acceptance criteria

1. A hybrid query whose highest dense result is obsolete still returns the
   relevant current evidence without an exception. Every returned ID must resolve
   to a current chunk; the obsolete ID must be absent.
2. Pruning scans every page of index entries, deletes all and only obsolete IDs,
   and waits for deletion to finish. A valid current entry must be preserved.
3. Tests use a temporary fictional corpus and fake dense/index responses. They
   do not connect to a provider, mutate a real vector index or consume API budget.
4. Existing retrieval, abstention, JD-isolation, evaluation and budget tests remain
   passing. Do not infer improved retrieval scores from these regression checks.

## Non-goals

- A new indexing algorithm, incremental embedding cache or production deployment.
- Authentication, multi-user isolation, upload quotas or concurrent-index safety.
- A semantic claim-support guarantee or new public/full-LLM evaluation scores.

## Verification map

- Criterion 1: `tests/test_index_consistency.py::test_hybrid_search_ignores_obsolete_dense_ids`.
- Criterion 2: `tests/test_index_consistency.py::test_pruning_removes_only_obsolete_ids_across_pages`.
- Criterion 3: fake responses and `build_workflow(tmp_path)` with no provider key.
- Criterion 4: run `python -m pytest` from the repository root.

These checks cover deterministic behavior around stale entries. They do not
exercise a live Qdrant service, concurrent rebuilds or provider failures.
