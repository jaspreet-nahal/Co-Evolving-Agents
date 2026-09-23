# Legacy Architecture Decisions

## D-L1: Fixed six-stage pipeline

**Status:** Implemented

Use one planner, retrieval, working-memory, sufficiency, synthesis, and verifier sequence for every benchmark.

## D-L2: Benchmark adapters configure task behavior

**Status:** Implemented

Adapters own dataset loading, corpus loading, task scoring, and per-example sufficiency settings while the core pipeline remains shared.

## D-L3: CorpusIndex is the replacement boundary

**Status:** Implemented

Search, grep, document lookup, and chunk lookup are exposed through `CorpusIndex`; `InMemoryCorpusIndex` is the reference implementation.

## D-L4: Track seen chunks

**Status:** Implemented

`SearchReadTools` stores `seen_chunk_ids` and excludes them from later search calls within a trajectory.

## D-L5: Keep evidence memory separate

**Status:** Implemented

`WorkingMemory` stores retrieved evidence and provenance independently of the corpus index.

## D-L6: Log sufficiency independently

**Status:** Implemented with limitation

Sufficiency is a distinct stage for diagnostics, but a negative result does not initiate another retrieval pass.

## D-L7: Require citations

**Status:** Implemented

Synthesized claims include chunk IDs and the verifier checks citation existence and token overlap.

## D-L8: Use deterministic core metrics

**Status:** Implemented

Trajectory recall and output recall distinguish evidence discovery from evidence used in the final answer.

## D-L9: Preserve simple local execution

**Status:** Implemented

The in-memory index and rule-based planner/synthesis path allow tests and sample benchmark runs without model hosting.

## D-L10: Known legacy limitations

**Status:** Limitation

The legacy executor is serial and linear, has no curation or episode graph, does not use all exposed tools as policy actions, and does not enforce `max_search_steps`.
