# Project Architecture

## 1. Purpose and Scope

This repository is a research evaluation harness for deep-research agent models. It supports both the original fixed six-stage workflow and an additive Harness-1 stateful action loop over multiple benchmarks.

The current implementation supports:

- A shared six-stage trajectory pipeline.
- Replaceable corpus indexes through the `CorpusIndex` interface.
- Benchmark-specific data loading, scoring, and sufficiency configuration.
- Per-stage JSON logs and per-trajectory summaries.
- Retrieval and answer citation recall metrics.
- Harness-1 episode state with candidate pools, curation, document review, verification cache, evidence links, and bounded observations.

The default implementation is rule-based. Harness-1 can use a deterministic policy or a lazy Hugging Face action policy configured for Gemma 4 and other compatible models.

## 2. Repository Layout

```text
.
|-- harness/
|   |-- core/
|   |   |-- models.py             # Shared dataclasses and enums
|   |   |-- harness.py            # Main trajectory and benchmark orchestration
|   |   |-- episode.py            # Harness-1 per-episode state
|   |   |-- actions.py            # Structured Harness-1 action contract
|   |   |-- observations.py       # Bounded WORKINGMEMORY rendering
|   |   |-- model_adapter.py      # Rule-based and Hugging Face policies
|   |   |-- planner.py            # Constraint extraction and query decomposition
|   |   |-- search_read.py        # Corpus contract, in-memory index, retrieval tools
|   |   |-- working_memory.py     # Evidence and provenance state
|   |   |-- sufficiency_check.py  # Benchmark-configurable evidence checks
|   |   |-- synthesis.py          # Rule-based answer and citation construction
|   |   |-- verifier.py            # Citation existence and token-overlap checks
|   |   |-- metrics.py             # Recall and failure-mode metrics
|   |   |-- logger.py              # JSON stage and result persistence
|   |   `-- __init__.py            # Public core API
|   |-- benchmarks/
|   |   |-- qampari.py
|   |   |-- browsecomp_plus.py
|   |   |-- financebench.py
|   |   |-- trec_biogen.py        # Verification-only stub
|   |   |-- freshstack.py          # Verification-only stub
|   |   `-- __init__.py            # Benchmark factories and exports
|   |-- tests/test_core.py         # Core unit and integration tests
|   `-- logs/                      # Runtime output, created as needed
|-- scripts/
|   |-- stage1_validate_qampari.py # Small QAMPARI validation workflow
|   `-- run_all_stages.py          # CLI dispatch for stages 1-5
|-- models/                        # Reserved for model integrations; currently empty
|-- qampari/                       # Vendored/reference QAMPARI material
|-- architecture.md
|-- architecture.mmd
|-- decisions.md
|-- requirements.txt
`-- pyproject.toml
```

## 3. Runtime Architecture

The runtime has two layers and two execution modes:

1. **Benchmark layer** loads examples and corpora, configures the harness for a task, runs trajectories, and applies task-specific scoring.
2. **Core layer** executes either the legacy six-stage pipeline or the Harness-1 policy/action loop and emits common trajectory data, logs, and recall metrics.

Execution modes are `legacy` through `run_trajectory()` and `harness1` through `run_harness1_episode()`. Harness-1 still emits the six compatibility stage records.

The executable entry point for multi-stage evaluation is `scripts/run_all_stages.py`. The reusable programmatic entry point is `DeepResearchHarness.run_harness1_episode`; `run_benchmark` remains available for generic query dictionaries.

## 4.1 Harness-1 State Machine

Harness-1 maintains an `EpisodeState` per query. The policy chooses semantic actions; the environment performs bookkeeping and deterministic state transitions.

State includes a candidate pool, an importance-tagged curated set capped at 30 by default, a full document store for review, a lightweight entity/year evidence graph, a claim/document verification cache, ordered action history, search history, turn budget, termination state, and duplicate accounting.

Supported actions are `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, and `end_search`. The first successful search can auto-seed up to eight fair candidates. `ObservationRenderer` exposes a bounded `WORKINGMEMORY` containing actionable state without inlining the full corpus.

## 4. End-to-End Control Flow

For each query, `DeepResearchHarness.run_trajectory`:

1. Creates a `Trajectory` containing the query, benchmark, model name, and optional gold chunk IDs.
2. Starts a trajectory logger and resets search deduplication and working memory.
3. Runs the planner stage.
4. Runs search/read for every planned sub-query.
5. Records the resulting working-memory state.
6. Checks configured sufficiency criteria.
7. Synthesizes an answer and chunk citations.
8. Verifies citations and claim support.
9. Computes trajectory recall, output recall, and a failure mode.
10. Saves the trajectory summary. Exceptions are recorded as an `ERROR:` answer and still produce a summary.

The pipeline is linear. The sufficiency result is logged and exposed on the trajectory, but it does not currently trigger another retrieval pass or stop synthesis.

## 5. The Six Core Stages

### 5.1 Planner

`Planner.plan` produces a `Plan` with constraints and sub-queries.

- Rule-based constraint extraction recognizes temporal, inclusion, exclusion, and quantity patterns.
- Query decomposition splits on question marks, `and`, `or`, and comma boundaries when useful.
- Answer type inference labels a sub-query as factual, list, count, boolean, or comparative.
- The optional LLM path currently falls back to the same rule-based planner.

The resulting constraints are copied into the public `Trajectory` model and logged.

### 5.2 Search/Read

`SearchReadTools` is an adapter over `CorpusIndex`.

Available tools:

- `search_corpus(query, top_k)`: ranked retrieval.
- `grep_corpus(pattern, max_results)`: regular-expression retrieval.
- `read_document(doc_id)`: returns all document chunks.
- `prune_chunks(chunk_ids)`: records a prune request for seen IDs.

The normal harness path calls `search_corpus` for each sub-query. It additionally calls `grep_corpus` when a query contains a capitalized term or a number/year. `read_document` and `prune_chunks` are public tools but are not invoked by the normal trajectory loop.

`SearchReadTools.seen_chunk_ids` is reset per trajectory and prevents a chunk from being returned again through the search/read tools.

### 5.3 Working Memory

`WorkingMemory` stores unique `EvidenceChunk` records keyed by chunk ID. Each record contains:

- The corpus chunk.
- Retrieval tool and query/pattern provenance.
- Retrieval step and relevance score.
- Pruning state, time, and reason.

Synthesis receives active, unpruned chunks sorted by relevance score. Working memory is separate from the corpus index so retrieval storage and answer context remain distinct concerns.

### 5.4 Sufficiency Check

`SufficiencyChecker` applies a benchmark-provided `SufficiencyConfig`. Available criteria are:

- At least one answer/evidence chunk.
- A minimum evidence-chunk count.
- Constraint satisfaction by term presence.
- A configured expected answer count.
- A confidence threshold.

The checker returns a boolean, explanation, missing-information list, and heuristic confidence. Its result is logged but is currently advisory: the pipeline continues to synthesis regardless of the decision.

### 5.5 Synthesis

`SynthesisEngine` creates `Claim` objects and a final answer from active working-memory chunks.

- List-like questions extract sentence-shaped list items containing capitalized terms.
- Other questions use up to five highest-scoring chunks as factual claims.
- Each claim cites one or more chunk IDs using `[chunk_id]` markers.
- Empty evidence produces an insufficient-evidence answer.
- The optional LLM path currently falls back to rule-based synthesis.

The synthesized answer and cited chunk IDs are written to the trajectory.

### 5.6 Verifier

`Verifier` checks each claim citation against working memory and the corpus index. It also computes token overlap between a claim and its cited chunk using the configured minimum overlap.

The verifier reports invalid citation IDs and per-claim support details. `all_citations_valid` is based on invalid citation IDs; a claim with an existing citation but insufficient token overlap is recorded as unsupported in `support_details` and does not itself invalidate the whole result.

## 6. Data Model and Boundaries

The principal data types are:

- `CorpusDocument`: source document and its chunks.
- `CorpusChunk`: retrievable unit with document ID, text, offsets, metadata, and optional embedding.
- `Trajectory`: complete query execution record.
- `StageLog`: structured input/output record for one stage.
- `EvidenceChunk`: working-memory wrapper with provenance.
- `BenchmarkResult`: aggregate result over trajectories.

`CorpusIndex` is the main replacement boundary. `InMemoryCorpusIndex` stores documents and chunks in dictionaries and implements token-overlap search plus regex grep. Its search score combines a deterministic overlap component with a small random component, so equal corpus/query runs are not strictly deterministic.

## 7. Benchmark Layer

All benchmark adapters expose a factory, data loading, corpus loading, and evaluation behavior where implemented.

| Benchmark | Current role | Data/evaluation behavior |
|---|---|---|
| QAMPARI | Multi-answer QA | Local/Hugging Face loading with sample fallback; precision, recall, F1, exact match, and exhaustiveness scoring. |
| BrowseComp-Plus | Hard browsing QA | Encrypted dataset loading/decryption path; sample fallback; exact-match scoring. Pre-built index loading is not implemented. |
| FinanceBench | Financial QA | Local/Hugging Face loading and corpus loading; extraction, calculation, and reasoning-oriented scoring. |
| TREC-Biogen | Biomedical generative retrieval | Verification-only stub; data access, topics, corpus, qrels, and metrics must be confirmed before implementation. |
| FreshStack | Procedural technical reasoning | Verification-only stub; the correct dataset split and format must be identified before implementation. |

Benchmark adapters replace the harness sufficiency configuration per example when task requirements differ. They also calculate task-specific metrics in addition to the shared trajectory/output recall values.

## 8. Persistence and Observability

`StageLogger` writes JSON under:

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

Benchmark aggregates are written under `harness/logs/{benchmark}/{model_name}/` as a timestamped result and `latest_results.json`. `ConsoleLogger` provides human-readable progress but is not the source of record.

## 9. Metrics

For a trajectory with gold-relevant chunks:

- **Trajectory recall** = gold chunks encountered during retrieval / total gold chunks.
- **Output recall** = gold chunks cited in the final answer / total gold chunks.

The difference identifies whether relevant evidence was never retrieved or was retrieved but not used. Failure modes are `success`, `found_not_used`, `never_found`, `partial`, and `no_gold_standard`.

Benchmark-specific answer metrics remain separate because answer formats and task goals differ. The generic `BenchmarkResult` currently leaves sufficiency and citation aggregate accuracy at `0.0`; adapters compute their own task-specific aggregates.

## 10. Operational Workflows

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the focused validation:

```bash
python scripts/stage1_validate_qampari.py
```

Run a selected stage:

```bash
python scripts/run_all_stages.py --stage 1 --model test_model
python scripts/run_all_stages.py --stage 2 --model test_model
python scripts/run_all_stages.py --stage 3 --model test_model
python scripts/run_all_stages.py --stage 4 --verify-only
python scripts/run_all_stages.py --stage 5 --verify-only
```

The CLI requires stage 1 to pass in the same invocation before stages 2 and 3 are run. Stages 4 and 5 intentionally stop at access/split verification.

## 11. Current Constraints and Risks

- Harness components are mutable and reused; a single harness instance is not designed for concurrent trajectories.
- `HarnessConfig.max_search_steps` is present but the current loop processes all planned sub-queries and does not enforce the limit.
- The advertised four search/read tools are not all used by the default loop.
- LLM integration points are placeholders, not active model adapters.
- Several sample corpus builders pass dictionaries to `InMemoryCorpusIndex.add_document`, whose current contract expects a `CorpusDocument`; those fallback paths need normalization before they can be relied upon.
- The in-memory search score contains randomness and is not suitable as a production index.
- TREC-Biogen and FreshStack are not executable evaluations yet.
- Core tests cover the core components and one integrated trajectory, but not benchmark adapters, external data loading, logging persistence, or CLI workflows.

## 12. Architecture Diagram

The standalone Mermaid source is in [architecture.mmd](architecture.mmd). It describes the same control flow and boundaries documented above.
