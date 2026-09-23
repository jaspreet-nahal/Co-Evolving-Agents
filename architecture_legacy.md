# Legacy Architecture

## Scope

This document describes the original batch deep-research evaluation harness before Harness-1 stateful episodes were added. The legacy path remains available through `DeepResearchHarness.run_trajectory()` when `HarnessConfig.execution_mode` is `legacy`.

## Runtime Flow

Each query runs a fixed six-stage pipeline:

1. **Planner**: extracts rule-based constraints and decomposes the query into sub-queries.
2. **Search/Read**: calls `search_corpus` for each sub-query and conditionally calls `grep_corpus` for names or numeric patterns.
3. **Working Memory**: stores retrieved chunks and provenance in `WorkingMemory`.
4. **Sufficiency Check**: applies benchmark-specific evidence, constraint, answer-count, and confidence criteria.
5. **Synthesis**: creates factual or list claims from active chunks and emits `[chunk_id]` citations.
6. **Verifier**: validates cited chunk IDs and measures token overlap between claims and evidence.

After the six stages, the harness computes trajectory recall and output recall, then persists a trajectory summary.

## Main Components

- `harness/core/harness.py`: orchestration and legacy trajectory lifecycle.
- `harness/core/planner.py`: constraint extraction and sub-query planning.
- `harness/core/search_read.py`: `CorpusIndex`, `InMemoryCorpusIndex`, and retrieval tools.
- `harness/core/working_memory.py`: evidence chunks, provenance, and pruning state.
- `harness/core/sufficiency_check.py`: configurable sufficiency rules.
- `harness/core/synthesis.py`: rule-based claim and answer generation.
- `harness/core/verifier.py`: citation existence and support checks.
- `harness/core/metrics.py`: recall and failure-mode metrics.
- `harness/core/logger.py`: JSON stage logs and summaries.

## State and Boundaries

`SearchReadTools.seen_chunk_ids` prevents a chunk from being returned twice during one trajectory. `WorkingMemory` is separate from the global corpus index and is reset before every trajectory. The executor is serial and mutable; it is not designed for concurrent trajectories.

The corpus boundary is the abstract `CorpusIndex` interface. The reference implementation is an in-memory token-overlap index with regex grep. Documents can be loaded from JSONL or inserted as `CorpusDocument` objects.

## Limitations

- The pipeline is linear rather than an interactive policy loop.
- Sufficiency is logged but does not trigger additional retrieval.
- The default loop does not use `read_document` or `prune_chunks` as policy actions.
- There is no curated document set, importance ranking, evidence graph, or verification cache.
- Planner and synthesis LLM hooks currently fall back to rule-based behavior.
- `max_search_steps` is configured but not enforced by the legacy loop.
- Benchmark adapters calculate task-specific scores separately from generic recall metrics.

## Outputs

```text
harness/logs/{query_id}/
|-- planner/*.json
|-- search_read/*.json
|-- working_memory/*.json
|-- sufficiency_check/*.json
|-- synthesis/*.json
|-- verifier/*.json
`-- trajectory_summary.json
```
