# Harness-1 Architecture Decisions

## D-H1: Add Harness-1 without breaking legacy callers

**Status:** Implemented

Expose `run_harness1_episode()` and select it through `HarnessConfig(execution_mode="harness1")`, while preserving `run_trajectory()` and the six stage records.

## D-H2: Make state episode-local

**Status:** Implemented

Create a fresh `EpisodeState` per query so candidates, curation, document memory, graph, cache, history, and budget cannot leak between episodes.

## D-H3: Offload recoverable bookkeeping

**Status:** Implemented

The policy chooses semantic actions. The harness maintains IDs, provenance, deduplication, capacity, cache keys, state rendering, and termination accounting.

## D-H4: Use structured actions

**Status:** Implemented

Validate actions through `HarnessAction` and `ActionType` rather than relying on free-form transcript conventions.

## D-H5: Warm-start curation

**Status:** Implemented

Auto-seed up to eight candidates at fair importance after the first successful search, giving the policy a set to refine instead of forcing construction from an empty output.

## D-H6: Use importance-aware curation

**Status:** Implemented

Curated documents have four importance levels. The set is capped at 30 by default and evicts the lowest-importance item first.

## D-H7: Preserve full documents outside the prompt

**Status:** Implemented

The document store supports `read_document` and `review_docs` without repeatedly calling the corpus or placing all full text in every observation.

## D-H8: Render bounded working memory

**Status:** Implemented

`ObservationRenderer` prioritizes curated documents, recent candidates, graph bridges, verification, history, and budget within a configured character limit.

## D-H9: Cache verification

**Status:** Implemented

Verification results are keyed by claim and document ID so repeated checks reuse prior deterministic results.

## D-H10: Use a deterministic baseline policy

**Status:** Implemented

`RuleBasedPolicy` provides reproducible local behavior and a fallback for malformed model outputs.

## D-H11: Support Gemma 4 lazily

**Status:** Implemented

`TransformersActionPolicy` defaults to `google/gemma-4-31B-it` and imports/loads the model only when explicitly used. Runtime dependencies are declared separately from core logic.

## D-H12: Curated output controls synthesis

**Status:** Implemented

At episode completion, non-curated evidence is pruned from the synthesis context while trajectory recall still counts all encountered evidence.

## D-H13: Known Harness-1 limitations

**Status:** Limitation

The current evidence graph uses regex extraction, verification uses token overlap rather than a learned entailment verifier, and the in-memory retrieval index remains a reference implementation rather than a production hybrid retriever.
