# Architecture Decisions

This document records the design decisions visible in the current repository. Decisions marked **Implemented** are reflected in the code today. Decisions marked **Pending** or **Limitation** describe an intentional boundary that still requires work before it can be treated as production behavior.

The current design has two compatible execution modes: the original batch pipeline and the additive Harness-1 stateful action loop.

## 1. Evaluation Decisions

### D-001: Use one fixed harness across benchmarks

**Status:** Implemented

All benchmarks share the same six-stage core pipeline. Benchmark adapters may change data loading, scoring, and sufficiency criteria, but they do not fork the core reasoning flow. This makes cross-benchmark comparisons more meaningful.

### D-002: Treat each query as a trajectory

**Status:** Implemented

Every query receives a `Trajectory` containing its inputs, constraints, stage logs, retrieved chunks, cited chunks, answer, sufficiency decision, and recall metrics. This creates a consistent unit for debugging and aggregation.

### D-003: Separate retrieval quality from answer-use quality

**Status:** Implemented

The harness records both trajectory recall and output recall. Their difference exposes cases where evidence was retrieved but not cited, represented by the `found_not_used` failure mode.

### D-004: Keep benchmark scoring separate from shared metrics

**Status:** Implemented

QAMPARI uses answer-set precision/recall/F1 and exhaustiveness, BrowseComp-Plus uses exact match, and FinanceBench uses type-aware answer scoring. These remain benchmark adapter responsibilities instead of being forced into one generic score.

## 2. Pipeline Decisions

### D-005: Make constraints explicit

**Status:** Implemented

The planner extracts structured temporal, inclusion, exclusion, and quantity constraints instead of leaving them implicit in a prompt. Constraints are logged and can be checked by sufficiency logic.

### D-006: Separate planning from retrieval

**Status:** Implemented

The planner emits sub-queries, while `SearchReadTools` owns corpus access and retrieval deduplication. This keeps query interpretation independent from index implementation.

### D-007: Use a replaceable corpus index boundary

**Status:** Implemented

The abstract `CorpusIndex` contract isolates search, grep, document lookup, and chunk lookup. `InMemoryCorpusIndex` is the current test/reference implementation and can be replaced by a production index without changing the harness orchestration API.

### D-008: Prevent re-retrieval within a trajectory

**Status:** Implemented

`SearchReadTools` maintains `seen_chunk_ids` and passes them to the corpus index. This avoids repeatedly returning the same evidence and makes retrieval behavior inspectable.

### D-009: Keep working memory separate from the corpus index

**Status:** Implemented

The index owns the corpus; working memory owns the evidence selected for the current trajectory, including provenance, scores, and pruning state. This supports synthesis without mutating corpus state.

### D-010: Make sufficiency a distinct logged stage

**Status:** Implemented with a current limitation

Sufficiency is evaluated and logged independently from synthesis so the run can explain why evidence was considered adequate. The current pipeline does not use a negative decision to trigger more retrieval or prevent synthesis.

### D-011: Require chunk citations in synthesized claims

**Status:** Implemented

Claims carry chunk IDs and formatted answers include `[chunk_id]` markers. The verifier can therefore validate citation existence and measure claim-to-evidence token overlap.

### D-012: Prefer deterministic verification over model judgment

**Status:** Implemented

Citation existence and support are checked with corpus lookup and token overlap. This makes verification reproducible and independent of an LLM response.

## 3. Integration Decisions

### D-013: Keep model integrations behind optional hooks

**Status:** Partially implemented

Planner and synthesis accept an `llm_client` and have `use_llm` switches. The hooks preserve a future integration point, but the current `_plan_with_llm` and `_synthesize_with_llm` methods delegate to rule-based behavior. The model registry and `models/` package are not implemented.

### D-014: Support local, Hugging Face, and sample data paths

**Status:** Implemented for the first three adapters

QAMPARI, BrowseComp-Plus, and FinanceBench attempt local or Hugging Face loading and provide sample data/corpora for development. This permits core validation without requiring every external dataset to be available.

### D-015: Gate uncertain benchmarks behind verification

**Status:** Implemented

TREC-Biogen and FreshStack remain verification-only because access requirements, dataset formats, splits, and evaluation details have not been confirmed. The CLI exposes checklists rather than pretending those evaluations are runnable.

### D-016: Use benchmark-specific sufficiency configuration

**Status:** Implemented

Adapters replace sufficiency criteria per example where task demands differ. For example, QAMPARI can require an expected answer count, while FinanceBench changes evidence requirements by question type.

### D-026: Add Harness-1 as an additive execution mode

**Status:** Implemented

The stateful action loop is exposed through `run_harness1_episode()` while `run_trajectory()` remains available for existing benchmark consumers. This preserves the current six-stage trajectory contract and allows incremental adoption.

### D-027: Move recoverable search bookkeeping into episode state

**Status:** Implemented

`EpisodeState` owns candidates, curated documents, full document memory, evidence links, verification records, history, and budgets. The policy retains semantic choices through structured actions; the harness performs normalization, deduplication, capacity management, caching, and rendering.

### D-028: Use importance-aware curation with warm-start auto-seeding

**Status:** Implemented

The first successful search may seed up to eight fair candidates. The policy can then promote or remove documents, while the harness enforces capacity and evicts the lowest-importance item first.

### D-029: Keep observations bounded and state-oriented

**Status:** Implemented

`ObservationRenderer` renders compact working memory instead of appending all retrieved text to every model prompt. Full documents remain available through `read_document` and `review_docs`.

### D-030: Provide a lazy model adapter for Gemma 4

**Status:** Implemented

`TransformersActionPolicy` defaults to `google/gemma-4-31B-it`, loads `AutoProcessor` and `AutoModelForMultimodalLM` only when selected, and falls back to `AutoModelForCausalLM` where needed. Malformed model output falls back to the deterministic policy.

## 4. Observability Decisions

### D-017: Persist structured JSON logs

**Status:** Implemented

Each stage writes structured input, output, timing, and metadata data. Each trajectory also gets a summary, and each benchmark/model pair gets timestamped and latest aggregate results.

### D-018: Keep console output separate from persisted logs

**Status:** Implemented

`ConsoleLogger` provides progress for interactive runs, while `StageLogger` is the durable source for analysis and validation.

### D-019: Validate the core before scaling benchmark runs

**Status:** Implemented

The stage runner requires stage 1 validation before stages 2 and 3 in an `--all` run. This establishes a low-cost gate before expensive benchmark execution.

## 5. Known Decisions To Revisit

### D-020: Use an in-memory token-overlap reference index

**Status:** Limitation

The current index is intentionally simple for tests and local validation. It is not a production retrieval strategy: it uses token overlap, truncates context in logs, and adds a random score component. A production implementation should provide deterministic, indexed retrieval behind the same `CorpusIndex` contract.

### D-021: Keep the trajectory executor stateful

**Status:** Limitation

The harness reuses mutable planner, search tools, working memory, checker, synthesis, verifier, and current-trajectory fields. Resetting search and memory per trajectory is sufficient for the current serial workflow, but concurrency would require isolation or a per-trajectory execution context.

### D-022: Expose four tools but use only two by default

**Status:** Limitation

The Context-1-inspired API includes search, grep, document read, and pruning. The default orchestration uses search and conditional grep; read and prune remain extension points rather than active control-flow steps.

In Harness-1 mode, `read_document`, `review_docs`, `curate`, `verify`, and `end_search` are active structured actions, and `fan_out_search` is bounded to five queries.

### D-023: Retain configuration fields before full enforcement

**Status:** Limitation

`max_search_steps` is part of the public configuration and CLI setup, but the current search loop does not enforce it. This should be resolved before using the setting as a hard experimental control.

### D-024: Normalize adapter-to-index document construction

**Status:** Pending

Several sample and Hugging Face corpus paths pass dictionaries to `InMemoryCorpusIndex.add_document`, while the index currently expects a `CorpusDocument`. The adapter boundary needs one canonical conversion path before fallback or external corpus loading can be considered reliable.

### D-025: Expand test coverage at integration boundaries

**Status:** Pending

Core unit tests and one integrated trajectory exist. Benchmark adapters, external loading, CLI behavior, log persistence, model hooks, and verification-only workflows still need focused tests.
