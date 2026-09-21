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
from harness.benchmarks import create_qampari_benchmark


def main():
    print("=" * 60)
    print("STAGE 1 VALIDATION: QAMPARI Benchmark")
    print("=" * 60)

    print("\n1. Creating QAMPARI benchmark...")
    benchmark = create_qampari_benchmark()

    print("2. Loading sample data...")
    examples = benchmark.load_data(max_examples=20)
    print(f"   Loaded {len(examples)} examples")

    print("3. Loading corpus...")
    corpus_index = benchmark.load_corpus()
    print(f"   Corpus has {len(corpus_index.documents)} documents, {len(corpus_index.chunks)} chunks")

    print("4. Creating harness...")
    config = HarnessConfig(
        max_search_steps=5,
        max_chunks_per_search=10,
        sufficiency_config=SufficiencyConfig(
            criteria=[SufficiencyCriterion.ALL_ANSWERS, SufficiencyCriterion.EVIDENCE_THRESHOLD],
            expected_answer_count=5,
            min_evidence_chunks=3
        ),
        model_name="test_model"
    )
    harness = create_harness(corpus_index, config)

    print("\n5. Running evaluation on 5 questions...")
    results = benchmark.run_evaluation(harness, max_examples=5)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nTotal Examples: {results['total_examples']}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.3f}")
    print(f"Exhaustiveness Accuracy: {results['exhaustiveness_accuracy']:.3f}")
    print(f"Avg Precision: {results['avg_precision']:.3f}")
    print(f"Avg Recall: {results['avg_recall']:.3f}")
    print(f"Avg F1: {results['avg_f1']:.3f}")
    print(f"Avg Trajectory Recall: {results['avg_trajectory_recall']:.3f}")
    print(f"Avg Output Recall: {results['avg_output_recall']:.3f}")
    print(f"Avg Sufficiency Accuracy: {results['avg_sufficiency_accuracy']:.3f}")

    print("\n" + "=" * 60)
    print("FIELD VALIDATION")
    print("=" * 60)

    all_fields_present = True
    for i, ex in enumerate(results['per_example']):
        print(f"\nExample {i+1}: {ex['question_id']}")

        required_fields = [
            'question_id', 'question', 'gold_answers', 'predicted_answers',
            'exhaustiveness', 'trajectory_recall', 'output_recall',
            'sufficiency_decision', 'sufficiency_reason', 'final_answer'
        ]

        for field in required_fields:
            if field in ex and ex[field] is not None:
                print(f"  ✓ {field}: present")
            else:
                print(f"  ✗ {field}: MISSING")
                all_fields_present = False

        exhaustiveness = ex.get('exhaustiveness', {})
        exhaustiveness_fields = ['precision', 'recall', 'f1', 'exhaustiveness', 'exact_match',
                                 'num_predicted', 'num_gold', 'num_correct',
                                 'correct_answers', 'missing_answers', 'extra_answers']
        for field in exhaustiveness_fields:
            if field in exhaustiveness:
                print(f"  ✓ exhaustiveness.{field}: {exhaustiveness[field]}")
            else:
                print(f"  ✗ exhaustiveness.{field}: MISSING")
                all_fields_present = False

    print("\n" + "=" * 60)
    print("LOG FILE VALIDATION")
    print("=" * 60)

    import glob
    log_dirs = glob.glob("harness/logs/q_*/")
    print(f"Found {len(log_dirs)} trajectory log directories")

    for log_dir in log_dirs[:2]:
        stage_files = glob.glob(os.path.join(log_dir, "*/*.json"))
        stages = set(os.path.basename(os.path.dirname(f)) for f in stage_files)
        print(f"  {os.path.basename(log_dir)}: stages = {stages}")

        expected_stages = {'planner', 'search_read', 'working_memory', 'sufficiency_check', 'synthesis', 'verifier'}
        missing = expected_stages - stages
        if missing:
            print(f"    MISSING STAGES: {missing}")
            all_fields_present = False
        else:
            print(f"    ✓ All 6 stages logged")

    print("\n" + "=" * 60)
    if all_fields_present:
        print("✓ STAGE 1 VALIDATION PASSED")
        print("All logged fields correctly populated.")
        print("Ready to proceed to Stage 2 (BrowseComp-Plus).")
    else:
        print("✗ STAGE 1 VALIDATION FAILED")
        print("Some fields are missing. Fix before proceeding.")
    print("=" * 60)

    return all_fields_present


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)