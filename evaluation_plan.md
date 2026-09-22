# Evaluation Plan: 20-Cell First Phase

## Decision

The first phase runs 20 execution cells:

| Model | BrowseComp-Plus | TREC-Biogen | FinanceBench | QAMPARI | FreshStack |
|---|---:|---:|---:|---:|---:|
| Gemma-4-31B-it | OFF + ON | OFF + ON | OFF + ON | OFF + ON | OFF + ON |
| Qwen3.8-27B | OFF + ON | OFF + ON | OFF + ON | OFF + ON | OFF + ON |

This is 2 models x 5 benchmarks x 2 modes = 20 cells. Each model/benchmark pair remains matched, so assistance gain is estimable everywhere in the first phase:

`Assistance Gain = score(ON) - score(OFF)`

Muse-Glimmer-30B and Gemma-4-26B-A4B-it are deferred to the expansion phase. Running all four models in only one mode would produce 20 cells but would not measure the proposed harness effect.

## Execution gates

The 20-cell count is valid only when every benchmark has a verified corpus, task set, scorer, and reproducible split. TREC-Biogen and FreshStack are currently blocked by the access and split checks already present in the runner. They must not be replaced with unverified sample data while reporting the first-phase result.

The first executable gate is a small paired pilot on QAMPARI, BrowseComp-Plus, and FinanceBench for both selected models and both modes. The full 20-cell phase starts after the pilot confirms that ON and OFF use the same corpus, prompts, synthesis requirement, verifier, query sample, and trial policy.

## Measurement priority

### Now: required for the first 20 cells

| Priority | Measurements |
|---|---|
| P0 | Task success or benchmark accuracy; assistance gain; mandatory requirement compliance; trial count and confidence intervals |
| P0 | Trajectory recall; output or curated-set recall where the benchmark has a valid gold set; claim coverage when claims can be aligned reproducibly |
| P0 | Search calls, read calls, unique sources, repeated actions, total turns, total duration, time to first relevant evidence, time to sufficiency |
| P0 | Primary failure category, secondary failure categories, first failure turn, recovery turn, and final failure category |
| P1 | Task descriptors: conceptual breadth, logical nesting, exploration, depth requirement, and coverage requirement |
| P1 | Failure-distribution shift and paired trajectory deltas between ON and OFF |

The first report should present per-pair ON, OFF, and gain values, followed by pooled estimates stratified by benchmark and model. It should not rank models using pooled scores without showing the paired mode contrast.

### Later: add after early signals are stable

| Priority | Promotion condition | Measurements |
|---|---|---|
| P2 | The first-phase logs have stable claim boundaries and source identifiers | Claim/subclaim ledger; direct, weak, missing, and conflicting support status; claim-to-evidence coverage |
| P2 | Citation extraction is reproducible across all five benchmark formats | Citation association/support and citation accuracy as separate measures |
| P2 | Enough trajectories exist to estimate rare behaviors reliably | Search branching, backtracking, efficiency decomposition, and recovery analysis by failure type |
| P2 | Manual annotation agreement is acceptable on a held-out trace subset | Expanded multi-label failure taxonomy and adjudicated taxonomy statistics |
| P3 | The first 20-cell result identifies benchmark or model interactions worth resolving | Muse-Glimmer-30B and Gemma-4-26B-A4B-it; additional trials; sensitivity analyses |

DRIFT-derived claim support labels and first/recovery-turn analysis remain supporting measurements, not publication-verified core evidence. CoEvoSkills, hard-negative generation, Agentic Evolution, RL, fine-tuning, self-evolution, and new benchmarks remain outside this phase.

## Reporting rule

The primary table has one row per model/benchmark pair and columns for OFF score, ON score, assistance gain, and trial count. A second table reports the ON-minus-OFF shift for each failure label and process metric. Missing benchmark access is reported as not run, never as a zero score.
