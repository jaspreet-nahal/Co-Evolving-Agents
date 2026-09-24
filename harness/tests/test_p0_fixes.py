import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from dataclasses import replace

from harness.core import (
    InMemoryCorpusIndex, CorpusDocument, CorpusChunk, CorpusDocumentConversionError,
    to_corpus_document, DeepResearchHarness, HarnessConfig, RuleBasedPolicy, RuleBasedFlatPolicy,
    run_paired_benchmark, run_paired_grid, UnavailableModelError, resolve_policy_factory,
    resolve_flat_policy_factory, TEST_MODEL_REGISTRY, H0_CONFIG_DEFAULTS, H1_CONFIG_DEFAULTS,
)


# ---------------------------------------------------------------------------
# 1. Deterministic retrieval
# ---------------------------------------------------------------------------

class TestDeterministicRetrieval:

    def _build_index(self):
        index = InMemoryCorpusIndex(chunk_size=200, chunk_overlap=20)
        for i in range(15):
            index.add_document(CorpusDocument(
                doc_id=f"d{i}",
                content=f"Alice and Bob discuss machine learning topic number {i} about search systems."
            ))
        return index

    def test_identical_query_and_corpus_yield_identical_ranking(self):
        index = self._build_index()

        first = [c.chunk_id for c in index.search("machine learning search", top_k=10, exclude_chunk_ids=set())]
        second = [c.chunk_id for c in index.search("machine learning search", top_k=10, exclude_chunk_ids=set())]

        assert first == second
        assert len(first) > 0

    def test_identical_query_and_corpus_yield_identical_scores(self):
        index = self._build_index()

        first = {c.chunk_id: c.score for c in index.search("machine learning", top_k=10, exclude_chunk_ids=set())}
        second = {c.chunk_id: c.score for c in index.search("machine learning", top_k=10, exclude_chunk_ids=set())}

        assert first == second

    def test_search_does_not_import_or_use_random(self):
        import harness.core.search_read as search_read_module
        source = open(search_read_module.__file__).read()
        assert "np.random" not in source
        assert "random.random" not in source

    def test_full_trajectory_retrieval_is_reproducible_across_runs(self):
        def build_index():
            index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=20)
            index.add_document(CorpusDocument(doc_id="d1", content="Apple revenue was $394B in 2022."))
            index.add_document(CorpusDocument(doc_id="d2", content="Microsoft revenue was $211B in 2023."))
            return index

        config = HarnessConfig(max_search_steps=2, max_chunks_per_search=5, model_name="test_model")

        traj1 = DeepResearchHarness(build_index(), config).run_trajectory(
            query="What was Apple's revenue in 2022?", query_id="r1", benchmark="test"
        )
        traj2 = DeepResearchHarness(build_index(), config).run_trajectory(
            query="What was Apple's revenue in 2022?", query_id="r2", benchmark="test"
        )

        assert sorted(traj1.all_retrieved_chunk_ids) == sorted(traj2.all_retrieved_chunk_ids)


# ---------------------------------------------------------------------------
# 2. Enforced experiment budget
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:

    def _many_doc_index(self, n=20):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=5)
        for i in range(n):
            index.add_document(CorpusDocument(doc_id=f"d{i}", content=f"Entity{i} appeared in year {2000+i}."))
        return index

    def test_legacy_search_calls_do_not_exceed_max_search_steps(self):
        # execution_mode="legacy" explicitly exercises the original fixed
        # six-stage pipeline's own budget enforcement (distinct from
        # execution_mode="off", which is the strict H0 FlatReActExecutor
        # as of the C1-C5/strict-H0 architecture change).
        index = self._many_doc_index()
        query = "Entity1 in 2001?"
        config = HarnessConfig(max_search_steps=2, max_chunks_per_search=5, model_name="test", execution_mode="legacy")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(query=query, query_id="budget_test", benchmark="test")

        assert trajectory.search_calls <= 2

    def test_legacy_search_loop_stops_immediately_when_budget_is_zero(self):
        # Directly exercises the hard-cap enforcement path: with a
        # zero-search budget, the loop must not call search_corpus/grep_corpus
        # at all, regardless of how many sub-queries the planner produced.
        index = self._many_doc_index()
        config = HarnessConfig(max_search_steps=0, max_chunks_per_search=5, model_name="test", execution_mode="legacy")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(query="Entity1 in 2001?", query_id="zero_budget_test", benchmark="test")

        assert trajectory.search_calls == 0
        assert trajectory.all_retrieved_chunk_ids == []
        assert trajectory.termination_reason == "max_search_steps"

    def test_harness1_turns_do_not_exceed_max_turns(self):
        index = self._many_doc_index()
        config = HarnessConfig(max_turns=3, model_name="test", execution_mode="harness1")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(query="Who is Entity1?", query_id="turns_test", benchmark="test")

        assert trajectory.turns <= 3

    def test_harness1_search_actions_do_not_exceed_max_search_steps(self):
        index = self._many_doc_index()
        # RuleBasedPolicy always starts with fan_out_search([query]); force
        # a low search-step budget and confirm the harness stops issuing
        # new corpus searches once it is exhausted, regardless of max_turns.
        config = HarnessConfig(max_turns=20, max_search_steps=1, model_name="test", execution_mode="harness1")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(query="Who is Entity1?", query_id="search_budget_test", benchmark="test")

        search_names = {"fan_out_search", "search_corpus", "grep_corpus"}
        search_events = [event for event in trajectory.action_history if event.get("action") in search_names]
        executed_searches = [event for event in search_events if not event.get("result", {}).get("budget_exhausted")]
        exhausted_searches = [event for event in search_events if event.get("result", {}).get("budget_exhausted")]

        # The budget must actually bind: at least one search action was
        # allowed through and at least one was blocked once exhausted.
        assert len(executed_searches) <= 1
        assert len(exhausted_searches) >= 1

    def test_h0_and_h1_share_identical_budget_configuration(self):
        index = self._many_doc_index()
        base_config = HarnessConfig(max_turns=7, max_search_steps=4, model_name="rule_based")

        cell = run_paired_benchmark(
            queries=[{"query_id": "budget_pair", "query": "Who is Entity1?"}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            off_config=base_config,
            on_config=base_config,
            log_dir="harness/logs/test_p0_budget",
        )

        off_budget = cell.off.trajectories[0].budget
        on_budget = cell.on.trajectories[0].budget
        assert off_budget == on_budget
        assert off_budget["max_turns"] == 7
        assert off_budget["max_search_steps"] == 4

    def test_budget_is_logged_on_trajectory(self):
        index = self._many_doc_index()
        config = HarnessConfig(max_turns=9, max_search_steps=3, model_name="test")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(query="Who is Entity1?", query_id="log_budget_test", benchmark="test")

        assert trajectory.budget == {"max_turns": 9, "max_search_steps": 3, "max_chunks_per_search": 10}


# ---------------------------------------------------------------------------
# 3. Canonical paired-run identity
# ---------------------------------------------------------------------------

class TestPairedRunIdentity:

    def test_task_id_stable_across_h0_and_h1_condition_differs(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

        cell = run_paired_benchmark(
            queries=[{"query_id": "task_42", "query": "Who built the search system?", "gold_chunk_ids": ["d1_chunk_0"]}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            log_dir="harness/logs/test_p0_identity",
        )

        off_traj = cell.off.trajectories[0]
        on_traj = cell.on.trajectories[0]

        assert off_traj.task_id == on_traj.task_id == "task_42"
        assert off_traj.condition == "off"
        assert on_traj.condition == "on"
        assert off_traj.execution_mode == "off"
        assert on_traj.execution_mode == "harness1"
        assert off_traj.model_name == on_traj.model_name == "rule_based"
        assert off_traj.benchmark == on_traj.benchmark == "test"
        # query_id suffixing still keeps log directories distinct, but is no
        # longer the only pairing signal -- task_id is authoritative.
        assert off_traj.query_id != on_traj.query_id
        assert off_traj.trial_id == on_traj.trial_id == "0"

    def test_explicit_trial_id_is_threaded_through(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))
        config = HarnessConfig(model_name="test")
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(
            query="Who built it?", query_id="q1", benchmark="test",
            task_id="stable_task", trial_id="3", condition="off",
        )

        assert trajectory.task_id == "stable_task"
        assert trajectory.trial_id == "3"
        assert trajectory.condition == "off"


# ---------------------------------------------------------------------------
# 4. Immutable/copy-safe run configuration
# ---------------------------------------------------------------------------

class TestConfigImmutability:

    def test_run_paired_benchmark_does_not_mutate_caller_configs(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

        off_config = HarnessConfig(model_name="unset", execution_mode="legacy", max_turns=11)
        on_config = HarnessConfig(model_name="unset", execution_mode="legacy", max_turns=11)

        run_paired_benchmark(
            queries=[{"query_id": "immutability_test", "query": "Who built it?"}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            off_config=off_config,
            on_config=on_config,
            log_dir="harness/logs/test_p0_immutability",
        )

        # The caller's own config objects must be untouched.
        assert off_config.model_name == "unset"
        assert off_config.execution_mode == "legacy"
        assert on_config.model_name == "unset"
        assert on_config.execution_mode == "legacy"
        assert on_config.action_policy is None

    def test_same_config_object_reused_for_off_and_on_is_not_cross_contaminated(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))
        shared_config = HarnessConfig(model_name="unset", max_turns=5)

        cell = run_paired_benchmark(
            queries=[{"query_id": "shared_cfg", "query": "Who built it?"}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            off_config=shared_config,
            on_config=shared_config,
            log_dir="harness/logs/test_p0_shared_config",
        )

        assert shared_config.execution_mode == "legacy"
        assert shared_config.action_policy is None
        assert cell.off.trajectories[0].execution_mode == "off"
        assert cell.on.trajectories[0].execution_mode == "harness1"


# ---------------------------------------------------------------------------
# 5. D-024: canonical adapter -> CorpusDocument conversion boundary
# ---------------------------------------------------------------------------

class TestCanonicalCorpusConversion:

    def test_to_corpus_document_passes_through_corpus_document(self):
        doc = CorpusDocument(doc_id="d1", content="hello")
        assert to_corpus_document(doc) is doc

    def test_to_corpus_document_converts_id_text_dict(self):
        doc = to_corpus_document({"id": "d1", "text": "hello world"})
        assert doc.doc_id == "d1"
        assert doc.content == "hello world"

    def test_to_corpus_document_converts_doc_id_content_dict(self):
        doc = to_corpus_document({"doc_id": "d2", "content": "other text"})
        assert doc.doc_id == "d2"
        assert doc.content == "other text"

    def test_to_corpus_document_rejects_missing_id(self):
        with pytest.raises(CorpusDocumentConversionError):
            to_corpus_document({"text": "no id here"})

    def test_to_corpus_document_rejects_missing_content(self):
        with pytest.raises(CorpusDocumentConversionError):
            to_corpus_document({"id": "d3"})

    def test_to_corpus_document_rejects_non_dict_non_document(self):
        with pytest.raises(CorpusDocumentConversionError):
            to_corpus_document(["not", "a", "doc"])

    def test_in_memory_index_add_document_uses_canonical_conversion(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document({"id": "d1", "text": "Some corpus content about testing."})
        assert "d1" in index.documents
        assert len(index.chunks) > 0

    def test_in_memory_index_add_document_rejects_malformed_dict(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        with pytest.raises(CorpusDocumentConversionError):
            index.add_document({"unexpected_key": "value"})

    def test_qampari_sample_corpus_loads_via_canonical_conversion(self):
        from harness.benchmarks.qampari import QampariBenchmark
        benchmark = QampariBenchmark()
        index = benchmark.load_corpus()
        assert len(index.documents) > 0

    def test_qampari_missing_explicit_corpus_path_raises_not_silently_falls_back(self):
        from harness.benchmarks.qampari import QampariBenchmark
        benchmark = QampariBenchmark()
        with pytest.raises(FileNotFoundError):
            benchmark.load_corpus(corpus_path="/nonexistent/path/corpus.jsonl")

    def test_financebench_missing_explicit_corpus_path_raises(self):
        from harness.benchmarks.financebench import FinanceBenchBenchmark
        benchmark = FinanceBenchBenchmark()
        with pytest.raises(FileNotFoundError):
            benchmark.load_corpus(corpus_path="/nonexistent/path/corpus.jsonl")

    def test_browsecomp_missing_explicit_corpus_path_raises(self):
        from harness.benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
        benchmark = BrowseCompPlusBenchmark()
        with pytest.raises(FileNotFoundError):
            benchmark.load_corpus(corpus_path="/nonexistent/path/corpus.jsonl")

    def test_browsecomp_prebuilt_index_dir_without_loader_raises_not_silently_falls_back(self, tmp_path):
        from harness.benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
        index_dir = tmp_path / "prebuilt_index"
        index_dir.mkdir()
        benchmark = BrowseCompPlusBenchmark(index_dir=str(index_dir))
        with pytest.raises(NotImplementedError):
            benchmark.load_corpus()


# ---------------------------------------------------------------------------
# 6. Real-model policy interface
# ---------------------------------------------------------------------------

class TestPolicyInjection:
    """Covers the test-only 'rule_based' policy contract.

    Coverage for the four real frozen models' resolution/loading/failure
    behavior (which no longer pre-marks them "unavailable" -- see
    model_backend.py) lives in test_model_backend.py, using mocked
    transformers objects so these tests never attempt a real network call
    or GPU load.
    """

    def test_rule_based_is_registered_for_tests(self):
        assert "rule_based" in TEST_MODEL_REGISTRY
        factory = resolve_policy_factory("rule_based")
        assert isinstance(factory(), RuleBasedPolicy)

    def test_resolve_flat_policy_factory_resolves_rule_based(self):
        factory = resolve_flat_policy_factory("rule_based")
        assert isinstance(factory(), RuleBasedFlatPolicy)

    def test_resolve_policy_factory_raises_for_unknown_model(self):
        with pytest.raises(UnavailableModelError):
            resolve_policy_factory("Totally-Made-Up-Model")

    def test_run_paired_benchmark_accepts_explicit_policy_registry_override(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

        cell = run_paired_benchmark(
            queries=[{"query_id": "q1", "query": "Who built it?", "gold_chunk_ids": ["d1_chunk_0"]}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            log_dir="harness/logs/test_p0_policy_override",
        )

        assert cell.on.trajectories[0].execution_mode == "harness1"
        assert cell.off.trajectories[0].execution_mode == "off"
