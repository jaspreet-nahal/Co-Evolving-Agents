import pytest

from harness.core import ActionType, CorpusDocument, DeepResearchHarness, EpisodeState, HarnessAction, HarnessConfig, InMemoryCorpusIndex, ObservationRenderer, run_paired_benchmark


def test_episode_state_curation_is_deterministic_and_bounded():
    state = EpisodeState("query", max_curated_docs=1)
    state.candidates["d1"] = state.candidates["d2"] = None
    state.candidates["d1"] = type("Candidate", (), {"item_id": "d1"})()
    state.candidates["d2"] = type("Candidate", (), {"item_id": "d2"})()

    first = state.curate(["d1"], [], {"d1": "low"})
    second = state.curate(["d2"], [], {"d2": "fair"})

    assert first["added"] == ["d1"]
    assert second["added"] == ["d2"]
    assert state.ordered_curated_ids() == ["d2"]


def test_actions_reject_unknown_names():
    with pytest.raises(ValueError):
        HarnessAction.from_dict({"action": "unknown"})

    action = HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "done"}})
    assert action.action == ActionType.END_SEARCH


def test_observation_renderer_is_bounded():
    state = EpisodeState("q", context_budget_chars=120)
    state.candidates["doc"] = type("Candidate", (), {"item_id": "doc", "doc_id": "doc", "snippet": "x" * 1000, "score": 1.0})()
    rendered = ObservationRenderer(120).render(state)
    assert len(rendered) <= 120
    assert rendered.startswith("WORKINGMEMORY")


def test_harness1_episode_records_actions_and_state():
    index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
    index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))
    harness = DeepResearchHarness(index, HarnessConfig(max_turns=5, model_name="test"))

    trajectory = harness.run_harness1_episode("Who built the search system?", query_id="h1_test", gold_chunk_ids=["d1_chunk_0"])

    assert trajectory.execution_mode == "harness1"
    assert trajectory.action_history
    assert trajectory.curated_document_ids == ["d1"]
    assert trajectory.termination_reason
    assert len(trajectory.stage_logs) == 6
    assert trajectory.completed_at is not None
    assert trajectory.verification["claims_verified"]
    assert trajectory.turns == len(trajectory.action_history)


def test_paired_benchmark_runs_same_query_in_both_modes():
    index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
    index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

    cell = run_paired_benchmark(
        queries=[{"query_id": "paired", "query": "Who built the search system?", "gold_chunk_ids": ["d1_chunk_0"]}],
        benchmark_name="test",
        corpus_index=index,
        model_name="test",
        log_dir="harness/logs/test_paired",
    )

    assert cell.off.trajectories[0].execution_mode == "off"
    assert cell.on.trajectories[0].execution_mode == "harness1"
    assert cell.assistance_gain == cell.on.avg_output_recall - cell.off.avg_output_recall
    assert sum(cell.failure_distribution_delta.values()) == 0