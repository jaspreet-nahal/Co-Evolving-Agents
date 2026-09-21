import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness.core import (
    InMemoryCorpusIndex,
    DeepResearchHarness,
    HarnessConfig,
    SufficiencyConfig,
    SufficiencyCriterion
)


def main():
    print("=" * 60)
    print("HARNESS CORE DEMO: Single Trajectory")
    print("=" * 60)

    print("\n1. Building sample corpus...")
    index = InMemoryCorpusIndex(chunk_size=200, chunk_overlap=50)

    docs = [
        {"id": "wiki_apple", "text": "Apple Inc. is an American multinational technology company. In fiscal year 2022, Apple's total revenue was $394.33 billion. The iPhone generated $205.49 billion in revenue. Services revenue reached $78.13 billion."},
        {"id": "wiki_msft", "text": "Microsoft Corporation reported revenue of $211.92 billion for fiscal year 2023. Gross margin was 68.9%. Cloud revenue (Azure) was $110.36 billion. Operating income was $88.52 billion."},
        {"id": "wiki_amzn", "text": "Amazon.com Inc. 2022 revenue was $513.98 billion. AWS revenue was $80.1 billion. R&D spending was $73.2 billion. Operating income was $12.25 billion."},
    ]

    for doc in docs:
        index.add_document(doc)

    print(f"   Corpus: {len(index.documents)} docs, {len(index.chunks)} chunks")

    print("\n2. Configuring harness...")
    config = HarnessConfig(
        max_search_steps=3,
        max_chunks_per_search=5,
        sufficiency_config=SufficiencyConfig(
            criteria=[SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.EVIDENCE_THRESHOLD],
            min_evidence_chunks=1
        ),
        model_name="demo_model"
    )

    harness = DeepResearchHarness(index, config)

    print("\n3. Running trajectory...")
    query = "What was Apple's revenue in 2022?"
    trajectory = harness.run_trajectory(
        query=query,
        query_id="demo_001",
        gold_chunk_ids=["wiki_apple_chunk_0"],
        benchmark="demo"
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nQuery: {trajectory.query}")
    print(f"Model: {trajectory.model_name}")
    print(f"Benchmark: {trajectory.benchmark}")

    print(f"\n--- Constraints Extracted ---")
    for c in trajectory.constraints:
        print(f"  [{c.type}] {c.description}")

    print(f"\n--- Sufficiency Check ---")
    print(f"  Sufficient: {trajectory.sufficiency_decision}")
    print(f"  Reason: {trajectory.sufficiency_reason}")

    print(f"\n--- Final Answer ---")
    print(f"  {trajectory.final_answer}")

    print(f"\n--- Citations ---")
    for cid in trajectory.final_cited_chunk_ids:
        print(f"  {cid}")

    print(f"\n--- Metrics ---")
    print(f"  Trajectory Recall: {trajectory.trajectory_recall:.3f}")
    print(f"  Output Recall: {trajectory.output_recall:.3f}")
    print(f"  Failure Mode: ", end="")

    if trajectory.trajectory_recall == 1.0 and trajectory.output_recall == 1.0:
        print("success")
    elif trajectory.trajectory_recall > 0.5 and trajectory.output_recall < trajectory.trajectory_recall:
        print("found_not_used (found but didn't cite)")
    elif trajectory.trajectory_recall < 0.5:
        print("never_found")
    else:
        print("partial")

    print(f"\n--- Stage Logs ---")
    for log in trajectory.stage_logs:
        print(f"  {log.stage.value}: {log.duration_ms:.1f}ms")

    print("\n" + "=" * 60)
    print("Demo complete! Check harness/logs/ for detailed logs.")
    print("=" * 60)


if __name__ == "__main__":
    main()