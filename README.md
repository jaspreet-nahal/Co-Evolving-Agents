# Deep Research Agent Evaluation Harness

A comprehensive evaluation framework for testing deep-research agent models across multiple benchmarks, using a fixed agentic RAG harness with detailed logging and metrics.

## Overview

This harness evaluates four frozen models across five benchmarks:
- **Models**: Muse-Glimmer-30B, Qwen3.8-27B, Gemma-4-26B-A4B-it, Gemma-4-31B-it
- **Benchmarks**: QAMPARI, BrowseComp-Plus, FinanceBench, TREC-Biogen, FreshStack

The harness implements a **fixed 6-stage loop** (built once, reused across all benchmarks):

1. **Planner** — Decomposes query, extracts constraints as structured list
2. **Search/Read** — Four tools modeled on Chroma's Context-1 design
3. **Working Memory** — Accumulates evidence chunks with provenance
4. **Sufficiency Check** — Explicit check: enough evidence for answer?
5. **Synthesis** — Produces answer with chunk ID citations
6. **Verifier** — Deterministic citation validation

It also provides an additive **Harness-1 stateful mode**. Set `HarnessConfig(execution_mode="harness1")` to run structured actions over episode-local candidate pools, importance-tagged curation, full-document review memory, evidence links, verification cache, and bounded `WORKINGMEMORY` observations. The default action policy is deterministic; `TransformersActionPolicy` can load `google/gemma-4-31B-it` lazily or another compatible Hugging Face model.

## Key Metrics (per Context-1 Report)

- **Trajectory Recall**: Fraction of gold-relevant chunks encountered at any point
- **Output Recall**: Fraction of gold-relevant chunks cited in final answer
- **Failure Mode Detection**: "found but didn't use" vs "never found"

## Project Structure

```
harness/
├── core/                    # Harness core (6 stages)
│   ├── models.py           # Data models (Trajectory, Chunk, etc.)
│   ├── logger.py           # Stage logging
│   ├── planner.py          # Stage 1: Query decomposition + constraints
│   ├── search_read.py      # Stage 2: 4 Context-1 tools
│   ├── working_memory.py   # Stage 3: Evidence accumulation
│   ├── sufficiency_check.py# Stage 4: Explicit sufficiency check
│   ├── synthesis.py        # Stage 5: Answer synthesis with citations
│   ├── verifier.py         # Stage 6: Citation verification
│   ├── metrics.py          # Trajectory/Output recall
│   └── harness.py          # Main orchestration
├── benchmarks/              # Benchmark adapters
│   ├── qampari.py          # Stage 1: Multi-answer QA (READY)
│   ├── browsecomp_plus.py  # Stage 2: Hard browsing (READY)
│   ├── financebench.py     # Stage 3: Financial QA (READY)
│   ├── trec_biogen.py      # Stage 4: TREC 2024 BioGen (NEEDS VERIFICATION)
│   └── freshstack.py       # Stage 5: Procedural reasoning (NEEDS VERIFICATION)
└── utils/                   # Utilities
scripts/
├── stage1_validate_qampari.py  # Stage 1 validation
└── run_all_stages.py           # Main runner
```

## Stages

### Stage 0: Harness Core ✅ COMPLETE
All 6 stages implemented with:
- Separate logging per stage per trajectory
- Context-1 style search tools (search_corpus, grep_corpus, read_document, prune_chunks)
- Re-retrieval loop prevention (tracks all seen chunk IDs)
- Working memory with provenance tracking
- Explicit sufficiency check as separate logged field
- Synthesis with mandatory chunk citations
- Verifier with deterministic checks
- Trajectory/Output recall metrics

### Stage 1: QAMPARI ✅ READY FOR VALIDATION
- **Repo**: https://github.com/samsam3232/qampari
- **Paper**: https://arxiv.org/abs/2205.12665
- **Task**: Multi-answer questions requiring multiple paragraphs
- **Key Feature**: Exhaustiveness scorer (checks complete answer SET)
- **Validation**: Run `python scripts/stage1_validate_qampari.py`

### Stage 2: BrowseComp-Plus ✅ READY
- **Code**: https://github.com/texttron/BrowseComp-Plus
- **Data**: https://huggingface.co/datasets/Tevatron/browsecomp-plus (obfuscated, needs decrypt)
- **Corpus**: https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus (~100K docs)
- **Indexes**: https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes (pre-built BM25 + Qwen3)
- **Note**: Full run with frontier model costs ~$1000 for 830 queries

### Stage 3: FinanceBench ✅ READY
- **Repo**: https://github.com/patronus-ai/financebench
- **Data**: https://huggingface.co/datasets/PatronusAI/financebench
- **Scope**: Only 150 of 10,231 questions publicly available
- **Task**: Financial document QA (extraction, reasoning, calculation)

### Stage 4: TREC-Biogen ⚠️ VERIFICATION REQUIRED
- **Track**: TREC 2024 Biomedical Generative Retrieval
- **Access**: Via NIST TREC (trec.nist.gov) - likely requires data-use agreement
- **Action**: Check `harness/benchmarks/trec_biogen.py` for verification checklist

### Stage 5: FreshStack ⚠️ VERIFICATION REQUIRED
- **Org**: https://huggingface.co/freshstack (Databricks/UWaterloo)
- **Project**: https://fresh-stack.github.io
- **Challenge**: Multiple splits by tech stack - must identify correct one(s)
- **Action**: Check `harness/benchmarks/freshstack.py` for verification checklist

## Installation

```bash
pip install -r requirements.txt
# Required: numpy, requests, tqdm, pyyaml, datasets, huggingface-hub
```

## Running

### Stage 1 Validation (Required First Step)
```bash
python scripts/stage1_validate_qampari.py
```
This validates all logged fields are populated correctly on a small sample (~20 questions).

### Run Specific Stage
```bash
# Stage 1
python scripts/run_all_stages.py --stage 1 --model muse_glimmer_30b

# Stage 2 (after Stage 1 passes)
python scripts/run_all_stages.py --stage 2 --model muse_glimmer_30b

# Stage 3
python scripts/run_all_stages.py --stage 3 --model muse_glimmer_30b
```

### Verify Stages 4-5 Access
```bash
python scripts/run_all_stages.py --stage 4 --verify-only
python scripts/run_all_stages.py --stage 5 --verify-only
```

## Log Output

Each trajectory creates a detailed log directory:
```
harness/logs/
└── {query_id}/
    ├── planner/*.json
    ├── search_read/*.json
    ├── working_memory/*.json
    ├── sufficiency_check/*.json
    ├── synthesis/*.json
    ├── verifier/*.json
    └── trajectory_summary.json
```

Benchmark results saved to:
```
harness/logs/{benchmark}/{model}/
├── results_{timestamp}.json
└── latest_results.json
```

## Key Design Decisions

1. **Fixed Harness**: Same 6-stage loop for all benchmarks - only sufficiency criteria change
2. **Explicit Constraints**: Planner extracts structured constraints, not implicit in prompts
3. **Re-retrieval Prevention**: Global seen_chunk_ids set prevents loops (Context-1 finding)
4. **Separate Sufficiency Log**: Distinct from final answer - enables "found but didn't use" detection
5. **Mandatory Citations**: Every claim must cite chunk IDs; Verifier checks deterministically
6. **Dual Recall Metrics**: Trajectory vs Output recall distinguishes retrieval vs synthesis failures

## Next Steps

1. ✅ Stage 0: Harness core built
2. 🔄 Stage 1: Validate QAMPARI on small sample
3. ⏳ Stage 2: Run BrowseComp-Plus (need decrypt + pre-built indexes)
4. ⏳ Stage 3: Run FinanceBench (150 public questions)
5. ⏳ Stage 4: Verify TREC-Biogen data access
6. ⏳ Stage 5: Verify FreshStack split selection
7. ⏳ Run all 4 models on validated benchmarks

## Evaluation Plan

### Purpose

The experiment asks one focused question:

> Does externalizing research state into a fixed harness improve deep-research performance compared with a flat transcript loop, when the model, benchmark, corpus, synthesis requirement, verifier, and trial policy are held constant?

The harness is the intended intervention. This is a measurement study only. It does not train models, fine-tune models, use reinforcement learning, evolve the harness, generate hard negatives, or add benchmarks.

### Two execution modes

Every matched model/benchmark pair has two runs:

| Mode | Description |
|---|---|
| OFF | Flat ReAct-style loop. The model manages its own search state in a growing transcript. There is no persistent harness memory structure. |
| ON | Harness-1 stateful loop. The harness maintains candidate documents, importance-tagged curation, evidence links, working memory, verification cache, and an explicit sufficiency check. |

Both modes use the same corpus, task examples, model, answer format, synthesis requirement, verifier, query sample, and number of trials. The mode is the only intended experimental variable.

### First phase: 20 execution cells

The first phase keeps the ON/OFF comparison statistically meaningful by running 10 matched pairs:

| Model | BrowseComp-Plus | TREC-Biogen | FinanceBench | QAMPARI | FreshStack |
|---|---:|---:|---:|---:|---:|
| Gemma-4-31B-it | OFF + ON | OFF + ON | OFF + ON | OFF + ON | OFF + ON |
| Qwen3.8-27B | OFF + ON | OFF + ON | OFF + ON | OFF + ON | OFF + ON |

This is 2 models x 5 benchmarks x 2 modes = 20 execution cells. Muse-Glimmer-30B and Gemma-4-26B-A4B-it are reserved for the expansion phase. Running all four models in only one mode would also create 20 cells, but it would not estimate harness assistance gain.

### Execution gates

The five benchmarks are:

- **QAMPARI**: multi-answer questions where exhaustiveness matters.
- **BrowseComp-Plus**: difficult open-web-style browsing and evidence discovery.
- **FinanceBench**: financial document extraction, reasoning, and calculation.
- **TREC-Biogen**: biomedical generative retrieval; access must be verified.
- **FreshStack**: procedural and technical reasoning; the correct split must be verified.

The first executable pilot uses QAMPARI, BrowseComp-Plus, and FinanceBench with both selected models and both modes. It must confirm that the paired runs use identical evaluation inputs and that logs are complete. TREC-Biogen and FreshStack enter the full first phase only after their corpus, task set, scorer, and reproducible split are verified. Missing access is reported as **not run**, never as a zero score.

### Primary result

For every model/benchmark pair, report:

`Assistance Gain = score(ON) - score(OFF)`

The primary table contains the OFF score, ON score, assistance gain, trial count, and uncertainty interval. Results are also pooled by model and by benchmark, but pooled rankings must not replace the paired comparison.

### Measurements required now

These are the first-phase measurements, ordered by importance:

| Priority | Measurements | Why they matter |
|---|---|---|
| P0 | Task success or benchmark accuracy | Final task outcome |
| P0 | Mandatory requirement compliance | Separates required conditions from optional answer quality |
| P0 | Trajectory recall and output or curated-set recall | Distinguishes evidence discovery from evidence preservation and use |
| P0 | Search calls, read calls, unique sources, repeated actions, total turns, and total duration | Basic process cost and activity |
| P0 | Time to first relevant evidence and time to sufficiency | Measures discovery and stopping efficiency |
| P0 | Primary and secondary failure categories | Shows what went wrong, allowing multiple labels when needed |
| P0 | First failure turn, recovery turn, and final failure category | Distinguishes prevention, recovery, and unrecovered failure |
| P1 | Conceptual breadth, logical nesting, exploration, depth requirement, and coverage requirement | Describes task difficulty without becoming additional performance scores |
| P1 | Failure-distribution shift and paired process deltas | Shows where ON differs from OFF, beyond final accuracy |

The eight-category failure taxonomy remains the project taxonomy. Its construction should use a manually annotated trace subset, annotation-consistency checks, and multi-label assignment where appropriate.

### Measurements for later phases

Add these only when the early logs make the required annotations reproducible:

- A claim/subclaim ledger with claim-to-evidence links.
- Claim coverage and support status: direct, weak, missing, or conflicting.
- Citation association/support and citation accuracy as separate measures.
- Search branching, backtracking, efficiency decomposition, and recovery by failure type once enough trajectories exist for stable estimates.
- Expanded adjudicated failure-taxonomy statistics after manual annotation agreement is acceptable.
- The two deferred models, additional trials, and sensitivity analyses after the first 20-cell result identifies useful model or benchmark interactions.

Claim-support labels and first/recovery-turn analysis are informed in part by DRIFT, which is treated as supporting arXiv work rather than publication-verified core evidence. The core measurement direction is informed by TRACE, Characterizing Deep Research, ResearchRubrics, DR-Arena, LiveResearchBench, DRBench, and the MAST taxonomy methodology. These works motivate measurements; they do not change the model set, benchmark set, or intervention.

### Interpretation

The goal is not only to determine whether ON scores higher. The analysis should identify whether any gain occurs in:

- discovery of relevant evidence;
- preservation and curation of evidence;
- synthesis and claim coverage;
- citation support and verification;
- search efficiency;
- depth or breadth of research; or
- prevention and recovery from failures.

A positive final score with worse process cost, or a neutral score with better evidence coverage and fewer unrecovered failures, should be reported as a meaningful tradeoff rather than collapsed into a single ranking.

The detailed phase checklist is maintained in [evaluation_plan.md](evaluation_plan.md).

## Citation

If you use this harness, please cite:
- Context-1: https://www.trychroma.com/research/context-1
- QAMPARI: https://arxiv.org/abs/2205.12665
- BrowseComp-Plus: https://github.com/texttron/BrowseComp-Plus
- FinanceBench: https://github.com/patronus-ai/financebench