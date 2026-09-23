# Harness-1 Architecture

## Scope

This document describes the stateful Harness-1 execution mode added to the deep-research coding agent. It is selected with `HarnessConfig(execution_mode="harness1")` or by calling `DeepResearchHarness.run_harness1_episode()`.

## Core Principle

The policy makes semantic search decisions. The harness maintains recoverable bookkeeping around those decisions. The policy chooses what to search, inspect, curate, verify, and when to stop; the environment maintains candidates, memory, curation, links, cache, history, and budget.

## Episode State

`EpisodeState` is created per query and contains:

- Candidate pool keyed by normalized document ID.
- Importance-tagged curated set with `very_high`, `high`, `fair`, and `low` levels.
- Curated capacity, defaulting to 30 documents, with lowest-importance eviction.
- Full document store for revisiting retrieved documents without a new corpus call.
- Evidence graph mapping extracted names and years to document IDs.
- Verification cache keyed by claim and document ID.
- Search and action history, turn counter, termination reason, and duplicate accounting.

## Structured Actions

The policy emits a validated `HarnessAction`:

- `fan_out_search`: run up to five queries.
- `search_corpus`: targeted corpus search.
- `grep_corpus`: exact or regex matching.
- `read_document`: load a full document into episode memory.
- `review_docs`: re-render documents already held in memory.
- `curate`: add, remove, promote, or demote candidates.
- `verify`: check a claim against selected remembered documents and cache the result.
- `end_search`: terminate and submit the curated document IDs.

The first successful search can auto-seed up to eight fair candidates. The policy then refines that tentative set.

## Observation Layer

`ObservationRenderer` builds a bounded `WORKINGMEMORY` observation containing:

- Query and turn budget.
- Importance-ordered curated set.
- Recent uncurated candidate pool.
- Evidence-graph bridges.
- Recent verification records.
- Recent action summaries.
- Latest tool result.

Full document text is kept outside the prompt-facing state and is available through `read_document` and `review_docs`. This prevents the transcript from becoming the only memory representation.

## Execution Flow

```text
create EpisodeState
        |
render WORKINGMEMORY
        |
policy selects HarnessAction
        |
validate and execute action
        |
update candidates, memory, graph, cache, history, and budget
        |
render next observation
        |
end_search or max_turns
        |
restrict synthesis context to curated documents
        |
legacy six-stage result records + Harness-1 action metadata
```

## Model Integration

`RuleBasedPolicy` is the deterministic local policy used by tests. `TransformersActionPolicy` lazily loads `google/gemma-4-31B-it` through Hugging Face `AutoProcessor` and `AutoModelForMultimodalLM`, with a causal-LM fallback. Invalid model output falls back to the rule-based policy.

## Compatibility

Harness-1 preserves the existing `Trajectory` API and six stage records. It adds execution mode, termination reason, curated IDs, and action history. Legacy benchmark consumers can continue to use `run_trajectory()` unchanged.
