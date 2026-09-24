"""Architecture tests for the P0-B strict-H0 / independent-C1-C5 work.

These are architecture tests, not real benchmark runs: they exercise the
harness's control-flow guarantees (component isolation, budget parity,
paired identity) using small in-memory corpora and RuleBasedPolicy /
RuleBasedFlatPolicy, not any real frozen model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from harness.core import (
    InMemoryCorpusIndex, CorpusDocument, DeepResearchHarness, HarnessConfig,
    H0_CONFIG_DEFAULTS, H1_CONFIG_DEFAULTS, EpisodeState, ComponentUnavailableError,
    RuleBasedPolicy, RuleBasedFlatPolicy, FlatReActExecutor, FlatReActExecutorError,
    ActionType, HarnessAction, run_paired_benchmark, SufficiencyDecision,
    WorkingMemory, SufficiencyChecker,
)
from harness.core.components import CandidatePool, CuratedSet, EvidenceGraph, VerificationCache, SufficiencyController


def _two_doc_index():
    index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
    index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system in 2020."))
    index.add_document(CorpusDocument(doc_id="d2", content="Alice also founded a startup in 2021."))
    return index


class _ScriptedPolicy:
    """Deterministic scripted policy for exercising specific action sequences."""

    def __init__(self, actions):
        self.actions = actions
        self.i = 0

    def choose_action(self, observation, state):
        action = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return HarnessAction.from_dict(action)


# ---------------------------------------------------------------------------
# TASK 1 -- strict H0
# ---------------------------------------------------------------------------

class TestStrictH0:

    def test_h0_does_not_instantiate_working_memory_or_sufficiency_checker(self):
        import ast
        import inspect

        flat_module_source = inspect.getsource(sys.modules["harness.core.flat_react"])
        tree = ast.parse(flat_module_source)
        # Only real code references count -- module/class docstrings that
        # mention these names descriptively (to say H0 does NOT use them)
        # must not trip this check. Walk the AST and collect every Name/
        # Attribute identifier actually used in executable code.
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)

        forbidden = {"WorkingMemory", "SufficiencyChecker", "CandidatePool", "CuratedSet", "EvidenceGraph", "VerificationCache"}
        assert not (used_names & forbidden), f"FlatReActExecutor references forbidden names: {used_names & forbidden}"

    def test_h0_trajectory_has_no_candidate_curated_graph_cache_state(self):
        index = _two_doc_index()
        config = HarnessConfig(max_turns=6, max_search_steps=4, model_name="test", execution_mode="off", **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_state_check", benchmark="test")

        assert trajectory.component_fingerprint == {
            "enable_candidate_pool": False, "enable_curated_set": False,
            "enable_evidence_graph": False, "enable_verification_cache": False,
            "enable_sufficiency_check": False,
        }
        # H0 never produces a curated_document_ids list (there is no curated
        # set to produce one from) and never runs any sufficiency stage.
        assert trajectory.curated_document_ids == []
        assert trajectory.sufficiency_events == []
        assert trajectory.sufficiency_decision is False
        assert trajectory.sufficiency_reason == ""

    def test_h0_stage_logs_do_not_include_sufficiency_check_stage(self):
        index = _two_doc_index()
        config = HarnessConfig(max_turns=6, model_name="test", execution_mode="off", **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_no_stage", benchmark="test")

        stage_names = {log.stage.value for log in trajectory.stage_logs}
        assert "sufficiency_check" not in stage_names
        assert "working_memory" not in stage_names

    def test_h0_does_not_use_hidden_retrieval_deduplication(self):
        # Same query issued twice by a scripted policy: with suppression
        # off, both searches must return the same top result.
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=6, max_search_steps=5, model_name="test", execution_mode="off",
                                action_policy=_ScriptedPolicy(actions), **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_no_dedup", benchmark="test")

        first_results = trajectory.action_history[0]["result"]["new_chunk_ids"]
        second_results = trajectory.action_history[1]["result"]["new_chunk_ids"]
        assert first_results == second_results
        assert len(first_results) > 0

    def test_search_read_tools_suppression_is_explicitly_configurable(self):
        from harness.core.search_read import SearchReadTools
        index = _two_doc_index()
        suppressing = SearchReadTools(index, suppress_seen=True)
        not_suppressing = SearchReadTools(index, suppress_seen=False)

        first = suppressing.search_corpus("Alice", top_k=5)
        second = suppressing.search_corpus("Alice", top_k=5)
        assert second["new_chunk_ids"] == []  # suppressed

        first_ns = not_suppressing.search_corpus("Alice", top_k=5)
        second_ns = not_suppressing.search_corpus("Alice", top_k=5)
        assert first_ns["new_chunk_ids"] == second_ns["new_chunk_ids"]  # not suppressed
        assert len(second_ns["new_chunk_ids"]) > 0

    def test_h0_can_perform_multiple_search_read_turns(self):
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "read_document", "arguments": {"doc_id": "d1"}},
            {"action": "read_document", "arguments": {"doc_id": "d2"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=10, max_search_steps=5, model_name="test", execution_mode="off",
                                action_policy=_ScriptedPolicy(actions), **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_multi_turn", benchmark="test")

        turns_taken = [a["action"] for a in trajectory.action_history]
        assert turns_taken == ["search_corpus", "read_document", "read_document", "end_search"]

    def test_h0_stops_only_via_policy_end_search_or_hard_budget(self):
        # Policy that never ends: H0 must stop exactly at max_turns, not
        # through any implicit harness-side termination.
        index = _two_doc_index()

        class NeverEndsPolicy:
            def choose_action(self, observation, state):
                return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": "Alice"}})

        config = HarnessConfig(max_turns=4, max_search_steps=100, model_name="test", execution_mode="off",
                                action_policy=NeverEndsPolicy(), **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_hard_budget", benchmark="test")

        assert trajectory.termination_reason == "max_turns"
        assert len(trajectory.action_history) == 4

    def test_h0_disallows_structured_h1_only_actions(self):
        index = _two_doc_index()
        actions = [{"action": "curate", "arguments": {"add_ids": ["d1"]}}]
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="off",
                                action_policy=_ScriptedPolicy(actions), **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="h0_disallowed_action", benchmark="test")
        # The FlatReActExecutorError is caught by run_flat_h0_trajectory's
        # own error handling and surfaced as an ERROR answer, not silently
        # ignored or reinterpreted as a legal H0 action.
        assert trajectory.final_answer.startswith("ERROR:")

    def test_h0_and_h1_share_the_same_benchmark_verifier_contract(self):
        index = _two_doc_index()
        cell = run_paired_benchmark(
            queries=[{"query_id": "contract_check", "query": "Who is Alice?", "gold_chunk_ids": ["d1_chunk_0"]}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            log_dir="harness/logs/test_p0b_contract",
        )
        off_traj = cell.off.trajectories[0]
        on_traj = cell.on.trajectories[0]
        # Both conditions produce a verification stage log with the same shape.
        assert "all_citations_valid" in off_traj.verification
        assert "all_citations_valid" in on_traj.verification
        assert off_traj.gold_relevant_chunk_ids == on_traj.gold_relevant_chunk_ids == ["d1_chunk_0"]
        assert off_traj.benchmark == on_traj.benchmark == "test"


# ---------------------------------------------------------------------------
# TASK 2 -- independent C1-C5
# ---------------------------------------------------------------------------

class TestIndependentComponents:

    def test_all_five_flags_exist_on_harness_config(self):
        config = HarnessConfig()
        for flag in ["enable_candidate_pool", "enable_curated_set", "enable_evidence_graph",
                     "enable_verification_cache", "enable_sufficiency_check"]:
            assert hasattr(config, flag)

    def test_h0_defaults_are_all_off(self):
        assert H0_CONFIG_DEFAULTS == {
            "enable_candidate_pool": False, "enable_curated_set": False,
            "enable_evidence_graph": False, "enable_verification_cache": False,
            "enable_sufficiency_check": False,
        }

    def test_h1_defaults_are_all_on(self):
        assert H1_CONFIG_DEFAULTS == {
            "enable_candidate_pool": True, "enable_curated_set": True,
            "enable_evidence_graph": True, "enable_verification_cache": True,
            "enable_sufficiency_check": True,
        }

    @pytest.mark.parametrize("flags", [
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": True, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": True, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": True, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": True, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": True},
        {"enable_candidate_pool": True, "enable_curated_set": True, "enable_evidence_graph": True, "enable_verification_cache": True, "enable_sufficiency_check": True},
    ], ids=["00000", "10000", "01000", "00100", "00010", "00001", "11111"])
    def test_all_seven_named_configurations_execute_independently(self, flags):
        index = _two_doc_index()
        config = HarnessConfig(max_turns=8, max_search_steps=5, model_name="test", execution_mode="harness1", **flags)
        harness = DeepResearchHarness(index, config)
        combo_id = "".join("1" if v else "0" for v in flags.values())
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id=f"combo_{combo_id}", benchmark="test")

        assert trajectory.completed_at is not None
        assert not trajectory.final_answer.startswith("ERROR:")
        expected_fingerprint = {
            "candidate_pool": flags["enable_candidate_pool"],
            "curated_set": flags["enable_curated_set"],
            "evidence_graph": flags["enable_evidence_graph"],
            "verification_cache": flags["enable_verification_cache"],
            "sufficiency_check": flags["enable_sufficiency_check"],
        }
        assert trajectory.component_fingerprint == expected_fingerprint

    def test_c1_off_has_no_candidate_pool_state(self):
        state = EpisodeState("q", enable_candidate_pool=False)
        assert state.candidate_pool is None
        with pytest.raises(ComponentUnavailableError):
            state.require_candidate_pool()

    def test_c2_off_has_no_curated_set_state_and_curate_unavailable(self):
        state = EpisodeState("q", enable_candidate_pool=True, enable_curated_set=False)
        assert state.curated_set is None
        with pytest.raises(ComponentUnavailableError):
            state.curate(["x"], [], {})

    def test_c2_disabled_when_c1_disabled_even_if_requested_enabled(self):
        # CuratedSet structurally depends on a CandidatePool; if C1 is off,
        # C2 cannot exist either, regardless of its own flag.
        state = EpisodeState("q", enable_candidate_pool=False, enable_curated_set=True)
        assert state.curated_set is None

    def test_c3_off_has_no_graph_state_and_graph_ops_unavailable(self):
        state = EpisodeState("q", enable_evidence_graph=False)
        assert state.evidence_graph is None
        with pytest.raises(ComponentUnavailableError):
            state.require_evidence_graph()

    def test_c4_off_verification_still_works_without_cache(self):
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "verify", "arguments": {"claim": "Alice built a search system", "doc_ids": ["d1"]}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=6, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions),
                                enable_candidate_pool=True, enable_curated_set=False,
                                enable_evidence_graph=False, enable_verification_cache=False,
                                enable_sufficiency_check=False)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="c4_off_verify", benchmark="test")

        verify_result = trajectory.action_history[1]["result"]
        assert verify_result["records"][0]["supported"] is True
        assert verify_result["records"][0]["cached"] is False

    def test_c5_off_sufficiency_state_does_not_run_and_check_unavailable(self):
        state = EpisodeState("q", enable_sufficiency_check=False)
        assert state.sufficiency is None
        with pytest.raises(ComponentUnavailableError):
            state.require_sufficiency()

    def test_search_ingestion_does_not_implicitly_activate_c1_c2_c3(self):
        # C1/C2/C3 all OFF: a search must not create any of their state.
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions),
                                enable_candidate_pool=False, enable_curated_set=False,
                                enable_evidence_graph=False, enable_verification_cache=True,
                                enable_sufficiency_check=False)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="no_implicit_activation", benchmark="test")

        assert trajectory.curated_document_ids == []  # no C2 state was created
        search_result = trajectory.action_history[0]["result"]
        assert search_result["new_documents"]  # the search itself succeeded

    def test_auto_seed_does_not_fire_when_c1_or_c2_disabled(self):
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        # C1 on, C2 off: auto-seed has nowhere to seed into.
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), auto_seed=True,
                                enable_candidate_pool=True, enable_curated_set=False,
                                enable_evidence_graph=False, enable_verification_cache=False,
                                enable_sufficiency_check=False)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="no_auto_seed", benchmark="test")
        assert trajectory.curated_document_ids == []

    def test_curated_context_pruning_is_noop_when_c2_disabled(self):
        # With C2 off, _apply_curated_context must leave working memory
        # untouched rather than pruning against a nonexistent curated set.
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions),
                                enable_candidate_pool=True, enable_curated_set=False,
                                enable_evidence_graph=False, enable_verification_cache=False,
                                enable_sufficiency_check=False)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="c2_off_no_prune", benchmark="test")
        assert len(trajectory.all_retrieved_chunk_ids) > 0


# ---------------------------------------------------------------------------
# TASK 3 -- real C3/C5 operations
# ---------------------------------------------------------------------------

class TestC3C5Operations:

    def test_graph_query_returns_documents_sharing_entity(self):
        index = _two_doc_index()
        actions = [
            {"action": "fan_out_search", "arguments": {"queries": ["Alice"]}},
            {"action": "graph_query", "arguments": {"entity": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=6, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), **H1_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="graph_query_test", benchmark="test")

        graph_result = trajectory.action_history[1]["result"]
        assert graph_result["tool"] == "graph_query"
        assert set(graph_result["documents"]) == {"d1", "d2"}

    def test_graph_neighbors_returns_related_documents(self):
        index = _two_doc_index()
        actions = [
            {"action": "fan_out_search", "arguments": {"queries": ["Alice"]}},
            {"action": "graph_neighbors", "arguments": {"doc_id": "d1"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=6, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), **H1_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="graph_neighbors_test", benchmark="test")

        neighbors_result = trajectory.action_history[1]["result"]
        assert neighbors_result["tool"] == "graph_neighbors"
        assert "d2" in neighbors_result["neighbors"]
        assert "d1" not in neighbors_result["neighbors"]

    def test_check_sufficiency_produces_explicit_search_again_or_answer_now(self):
        index = _two_doc_index()
        actions = [
            {"action": "check_sufficiency", "arguments": {}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), **H1_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="sufficiency_event_test", benchmark="test")

        decision = trajectory.action_history[0]["result"]["decision"]
        assert decision in (SufficiencyDecision.SEARCH_AGAIN.value, SufficiencyDecision.ANSWER_NOW.value)

    def test_check_sufficiency_events_are_logged_on_trajectory(self):
        index = _two_doc_index()
        actions = [
            {"action": "fan_out_search", "arguments": {"queries": ["Alice"]}},
            {"action": "check_sufficiency", "arguments": {}},
            {"action": "check_sufficiency", "arguments": {}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=8, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), **H1_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="sufficiency_log_test", benchmark="test")

        assert len(trajectory.sufficiency_events) == 2
        for event in trajectory.sufficiency_events:
            assert event["decision"] in (SufficiencyDecision.SEARCH_AGAIN.value, SufficiencyDecision.ANSWER_NOW.value)

    def test_sufficiency_is_usable_mid_trajectory_not_only_post_hoc(self):
        # A policy that checks sufficiency BEFORE deciding to search again
        # or stop, proving C5 is available during the loop.
        index = _two_doc_index()

        class SufficiencyDrivenPolicy:
            def __init__(self):
                self.checked = False

            def choose_action(self, observation, state):
                if state.turn == 0:
                    return HarnessAction.from_dict({"action": "fan_out_search", "arguments": {"queries": ["Alice"]}})
                if not self.checked:
                    self.checked = True
                    return HarnessAction.from_dict({"action": "check_sufficiency", "arguments": {}})
                last_action = state.action_history[-1]
                if last_action.action == "check_sufficiency" and last_action.result_summary.get("is_sufficient"):
                    return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "sufficient"}})
                return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "insufficient_but_budget"}})

        config = HarnessConfig(max_turns=8, model_name="test", execution_mode="harness1",
                                action_policy=SufficiencyDrivenPolicy(), **H1_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="mid_loop_sufficiency", benchmark="test")

        actions_taken = [a["action"] for a in trajectory.action_history]
        assert "check_sufficiency" in actions_taken
        assert actions_taken.index("check_sufficiency") < actions_taken.index("end_search")


# ---------------------------------------------------------------------------
# TASK 5 -- regression / scientific invariants
# ---------------------------------------------------------------------------

class TestScientificInvariants:

    def test_paired_h0_h1_same_model_task_corpus_budget(self):
        index = _two_doc_index()
        base_config = HarnessConfig(max_turns=6, max_search_steps=4)
        cell = run_paired_benchmark(
            queries=[{"query_id": "pair_check", "query": "Who is Alice?", "gold_chunk_ids": ["d1_chunk_0"]}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            off_config=base_config,
            on_config=base_config,
            log_dir="harness/logs/test_p0b_invariant_pairing",
        )
        off_traj = cell.off.trajectories[0]
        on_traj = cell.on.trajectories[0]
        assert off_traj.task_id == on_traj.task_id
        assert off_traj.model_name == on_traj.model_name
        assert off_traj.benchmark == on_traj.benchmark
        assert off_traj.budget == on_traj.budget

    def test_h0_has_no_structured_harness_state_invariant(self):
        index = _two_doc_index()
        config = HarnessConfig(max_turns=6, model_name="rule_based", execution_mode="off", **H0_CONFIG_DEFAULTS)
        harness = DeepResearchHarness(index, config)
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id="invariant_h0", benchmark="test")
        assert not any(trajectory.component_fingerprint.values())

    def test_h1_11111_has_all_five_components(self):
        state = EpisodeState("q", **H1_CONFIG_DEFAULTS)
        assert isinstance(state.candidate_pool, CandidatePool)
        assert isinstance(state.curated_set, CuratedSet)
        assert isinstance(state.evidence_graph, EvidenceGraph)
        assert isinstance(state.verification_cache, VerificationCache)
        assert isinstance(state.sufficiency, SufficiencyController)

    @pytest.mark.parametrize("flags", [
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": True, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": True, "enable_verification_cache": False, "enable_sufficiency_check": False},
        {"enable_candidate_pool": False, "enable_curated_set": False, "enable_evidence_graph": False, "enable_verification_cache": False, "enable_sufficiency_check": True},
        {"enable_candidate_pool": True, "enable_curated_set": True, "enable_evidence_graph": True, "enable_verification_cache": True, "enable_sufficiency_check": True},
    ], ids=["00000", "10000", "00100", "00001", "11111"])
    def test_disabled_components_produce_zero_activity(self, flags):
        index = _two_doc_index()
        actions = [
            {"action": "search_corpus", "arguments": {"query": "Alice"}},
            {"action": "end_search", "arguments": {"reason": "done"}},
        ]
        config = HarnessConfig(max_turns=5, model_name="test", execution_mode="harness1",
                                action_policy=_ScriptedPolicy(actions), **flags)
        harness = DeepResearchHarness(index, config)
        combo_id = "".join("1" if v else "0" for v in flags.values())
        trajectory = harness.run_trajectory(query="Who is Alice?", query_id=f"zero_activity_{combo_id}", benchmark="test")

        if not flags["enable_curated_set"]:
            assert trajectory.curated_document_ids == []
        if not flags["enable_sufficiency_check"]:
            assert trajectory.sufficiency_events == []

    def test_c1_c2_c3_do_not_activate_each_other_implicitly(self):
        # C1 on alone: C2 (curated_set) and C3 (evidence_graph) must not exist.
        state = EpisodeState("q", enable_candidate_pool=True, enable_curated_set=False, enable_evidence_graph=False)
        assert state.candidate_pool is not None
        assert state.curated_set is None
        assert state.evidence_graph is None

        # C3 on alone: C1 and C2 must not exist.
        state2 = EpisodeState("q", enable_candidate_pool=False, enable_curated_set=False, enable_evidence_graph=True)
        assert state2.evidence_graph is not None
        assert state2.candidate_pool is None
        assert state2.curated_set is None

    def test_c4_cache_is_independent_of_verification(self):
        cache = VerificationCache()
        assert cache.get("claim", "doc1") is None  # empty cache, no crash
        # Verification logic itself (token overlap) does not live in
        # VerificationCache at all -- it has no verify() method.
        assert not hasattr(cache, "verify")

    def test_c5_produces_explicit_search_again_and_answer_now_events(self):
        controller = SufficiencyController()
        wm = WorkingMemory()
        from harness.core.models import Trajectory
        trajectory = Trajectory(query_id="t", query="q", model_name="m", benchmark="b")

        insufficient_event = controller.check_sufficiency(wm, trajectory, [], turn=0)
        assert insufficient_event.decision == SufficiencyDecision.SEARCH_AGAIN.value

        index = _two_doc_index()
        chunk = list(index.chunks.values())[0]
        wm.add_chunks([chunk], "search", "query", [1.0])
        sufficient_event = controller.check_sufficiency(wm, trajectory, [], turn=1)
        assert sufficient_event.decision == SufficiencyDecision.ANSWER_NOW.value

        assert len(controller.history) == 2

    def test_same_retrieval_request_is_deterministic(self):
        index = _two_doc_index()
        first = [c.chunk_id for c in index.search("Alice", top_k=5, exclude_chunk_ids=set())]
        second = [c.chunk_id for c in index.search("Alice", top_k=5, exclude_chunk_ids=set())]
        assert first == second

    def test_budget_identical_across_h0_and_h1(self):
        index = _two_doc_index()
        base_config = HarnessConfig(max_turns=9, max_search_steps=6)
        cell = run_paired_benchmark(
            queries=[{"query_id": "budget_invariant", "query": "Who is Alice?"}],
            benchmark_name="test",
            corpus_index=index,
            model_name="rule_based",
            off_config=base_config,
            on_config=base_config,
            log_dir="harness/logs/test_p0b_budget_invariant",
        )
        assert cell.off.trajectories[0].budget == cell.on.trajectories[0].budget == {
            "max_turns": 9, "max_search_steps": 6, "max_chunks_per_search": 10,
        }

    def test_no_real_model_name_silently_resolves_to_rule_based(self):
        # Real models now resolve to a ModelBackedPolicy/ModelBackedFlatPolicy
        # (never to RuleBasedPolicy/RuleBasedFlatPolicy) -- see
        # test_model_backend.py for the full load-time failure-mode
        # coverage (auth/download/runtime errors), which is exercised with
        # mocked transformers objects, not a real download.
        from harness.core import resolve_policy_factory, resolve_flat_policy_factory, ModelBackedEpisodePolicy, ModelBackedFlatPolicy, RuleBasedPolicy, RuleBasedFlatPolicy
        for model_name in ["Muse-Glimmer-30B", "Qwen3.8-27B", "Gemma-4-26B-A4B-it", "Gemma-4-31B-it"]:
            policy = resolve_policy_factory(model_name)()
            flat_policy = resolve_flat_policy_factory(model_name)()
            assert isinstance(policy, ModelBackedEpisodePolicy)
            assert isinstance(flat_policy, ModelBackedFlatPolicy)
            assert not isinstance(policy, RuleBasedPolicy)
            assert not isinstance(flat_policy, RuleBasedFlatPolicy)
