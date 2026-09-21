import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness.core import (
    InMemoryCorpusIndex,
    DeepResearchHarness,
    HarnessConfig,
    create_harness,
    SufficiencyConfig,
    SufficiencyCriterion
)
from harness.benchmarks import (
    create_qampari_benchmark,
    create_browsecomp_plus_benchmark,
    create_financebench_benchmark,
    create_trec_biogen_benchmark,
    create_freshstack_benchmark,
    TREC_VERIFICATION_CHECKLIST,
    FRESHSTACK_VERIFICATION_CHECKLIST
)


def run_stage1_qampari(model_name: str, max_examples: int = 20):
    print("=" * 60)
    print("STAGE 1: QAMPARI Benchmark")
    print("=" * 60)

    benchmark = create_qampari_benchmark()
    examples = benchmark.load_data(max_examples=max_examples)
    corpus_index = benchmark.load_corpus()

    config = HarnessConfig(
        max_search_steps=5,
        max_chunks_per_search=10,
        sufficiency_config=SufficiencyConfig(
            criteria=[SufficiencyCriterion.ALL_ANSWERS, SufficiencyCriterion.EVIDENCE_THRESHOLD],
            expected_answer_count=5,
            min_evidence_chunks=3
        ),
        model_name=model_name
    )
    harness = create_harness(corpus_index, config)

    results = benchmark.run_evaluation(harness, max_examples=max_examples)

    print("\nStage 1 Results:")
    print(f"  Exhaustiveness Accuracy: {results['exhaustiveness_accuracy']:.3f}")
    print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
    print(f"  Avg F1: {results['avg_f1']:.3f}")
    print(f"  Avg Trajectory Recall: {results['avg_trajectory_recall']:.3f}")
    print(f"  Avg Output Recall: {results['avg_output_recall']:.3f}")

    if results['avg_trajectory_recall'] > 0 and results['avg_output_recall'] >= 0:
        print("\nStage 1 validation passed - all fields populated")
        return True
    else:
        print("\nStage 1 validation failed")
        return False


def run_stage2_browsecomp(model_name: str, max_examples: int = 20):
    print("=" * 60)
    print("STAGE 2: BrowseComp-Plus Benchmark")
    print("=" * 60)

    benchmark = create_browsecomp_plus_benchmark()
    examples = benchmark.load_data(max_examples=max_examples)
    corpus_index = benchmark.load_corpus()

    config = HarnessConfig(
        max_search_steps=10,
        max_chunks_per_search=10,
        sufficiency_config=SufficiencyConfig(
            criteria=[SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.CONSTRAINT_SATISFIED],
            min_evidence_chunks=2
        ),
        model_name=model_name
    )
    harness = create_harness(corpus_index, config)

    results = benchmark.run_evaluation(harness, max_examples=max_examples)

    print("\nStage 2 Results:")
    print(f"  Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
    print(f"  Avg Trajectory Recall: {results['avg_trajectory_recall']:.3f}")
    print(f"  Avg Output Recall: {results['avg_output_recall']:.3f}")

    return True


def run_stage3_financebench(model_name: str, max_examples: int = 20):
    print("=" * 60)
    print("STAGE 3: FinanceBench Benchmark (Public 150 Questions)")
    print("=" * 60)

    benchmark = create_financebench_benchmark()
    examples = benchmark.load_data(max_examples=min(max_examples, 150))
    corpus_index = benchmark.load_corpus()

    config = HarnessConfig(
        max_search_steps=8,
        max_chunks_per_search=15,
        sufficiency_config=SufficiencyConfig(
            criteria=[SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.EVIDENCE_THRESHOLD],
            min_evidence_chunks=2
        ),
        model_name=model_name
    )
    harness = create_harness(corpus_index, config)

    results = benchmark.run_evaluation(harness, max_examples=max_examples)

    print("\nStage 3 Results:")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.3f}")
    print(f"  Accuracy by Type: {results['accuracy_by_type']}")
    print(f"  Avg Trajectory Recall: {results['avg_trajectory_recall']:.3f}")
    print(f"  Avg Output Recall: {results['avg_output_recall']:.3f}")

    return True


def run_stage4_trec_biogen(model_name: str):
    print("=" * 60)
    print("STAGE 4: TREC-Biogen - VERIFICATION REQUIRED")
    print("=" * 60)

    benchmark = create_trec_biogen_benchmark()
    verification = benchmark.verify_data_access()

    print("Verification Results:")
    for key, value in verification.items():
        print(f"  {key}: {value}")

    print("\n" + TREC_VERIFICATION_CHECKLIST)
    print("Cannot proceed without data access verification.")
    return False


def run_stage5_freshstack(model_name: str):
    print("=" * 60)
    print("STAGE 5: FreshStack - VERIFICATION REQUIRED")
    print("=" * 60)

    benchmark = create_freshstack_benchmark()
    verification = benchmark.verify_data_access()

    print("Verification Results:")
    for key, value in verification.items():
        print(f"  {key}: {value}")

    print("\n" + FRESHSTACK_VERIFICATION_CHECKLIST)
    print("Cannot proceed without split identification.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run Deep Research Agent Evaluation Stages")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run specific stage")
    parser.add_argument("--all", action="store_true", help="Run all available stages")
    parser.add_argument("--model", type=str, default="test_model", help="Model name")
    parser.add_argument("--max-examples", type=int, default=20, help="Max examples per benchmark")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification for stages 4-5")

    args = parser.parse_args()

    stages_to_run = []

    if args.all:
        stages_to_run = [1, 2, 3, 4, 5]
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        stages_to_run = [1]

    print(f"Running stages: {stages_to_run}")
    print(f"Model: {args.model}")
    print(f"Max examples: {args.max_examples}")

    results = {}

    for stage in stages_to_run:
        if stage == 1:
            results[1] = run_stage1_qampari(args.model, args.max_examples)
        elif stage == 2:
            if not results.get(1):
                print("Stage 1 not passed. Run Stage 1 first.")
                continue
            results[2] = run_stage2_browsecomp(args.model, args.max_examples)
        elif stage == 3:
            if not results.get(1):
                print("Stage 1 not passed. Run Stage 1 first.")
                continue
            results[3] = run_stage3_financebench(args.model, args.max_examples)
        elif stage == 4:
            if args.verify_only:
                run_stage4_trec_biogen(args.model)
            else:
                print("Stage 4 requires verification first. Use --verify-only")
        elif stage == 5:
            if args.verify_only:
                run_stage5_freshstack(args.model)
            else:
                print("Stage 5 requires verification first. Use --verify-only")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stage, success in results.items():
        status = "PASSED" if success else "FAILED/SKIPPED"
        print(f"  Stage {stage}: {status}")


if __name__ == "__main__":
    main()