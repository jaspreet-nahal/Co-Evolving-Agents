from typing import Any, Optional

from .actions import ActionType, HarnessAction


class RuleBasedPolicy:
    """Deterministic H1 policy used by tests, not a frozen model.

    Must never be silently substituted for one of the four named frozen
    models in an actual experiment run (see evaluation.py's
    resolve_policy_factory, which enforces this).

    Adapts to whichever of C1/C2 are actually enabled on the EpisodeState
    it is given, so it works unmodified across the C1-C5 configurations
    that still route through the Harness-1 episode loop. It is not used
    for H0 -- H0 runs through FlatReActExecutor with RuleBasedFlatPolicy
    instead, since H0's policy sees a plain transcript view, not an
    EpisodeState with (possibly disabled) component slots.
    """

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        if state.turn == 0:
            return HarnessAction(ActionType.FAN_OUT_SEARCH, {"queries": [state.query]})

        if state.candidate_pool is not None and state.curated_set is not None:
            if state.candidate_pool.candidates and not state.curated_set.curated:
                ids = list(state.candidate_pool.candidates)[:8]
                return HarnessAction.from_dict({
                    "action": "curate",
                    "arguments": {"add_ids": ids, "importance": {item_id: "fair" for item_id in ids}},
                })
            unreviewed = [item_id for item_id in state.curated_set.ordered_curated_ids() if item_id not in state.document_store]
            if unreviewed:
                return HarnessAction.from_dict({"action": "read_document", "arguments": {"doc_id": unreviewed[0]}})
            if state.turn >= 3 or state.curated_set.curated:
                return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "rule_based_policy_complete"}})
            return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})

        if state.candidate_pool is not None:
            unreviewed = [item_id for item_id in state.candidate_pool.candidate_list() if item_id not in state.document_store]
            if unreviewed:
                return HarnessAction.from_dict({"action": "read_document", "arguments": {"doc_id": unreviewed[0]}})
            if state.turn >= 3 or state.candidate_pool.candidates:
                return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "rule_based_policy_complete"}})
            return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})

        # No candidate pool at all (C1 OFF): fall back to a plain
        # search-a-few-times-then-stop loop driven only by turn count and
        # documents already reviewed, with no structured state to inspect.
        if len(state.document_store) < 1 and state.turn < 2:
            return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})
        if state.turn >= 3:
            return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "rule_based_policy_complete"}})
        return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})


class RuleBasedFlatPolicy:
    """Deterministic H0 policy: reasons only from the flat transcript view.

    Not a frozen model -- reserved for tests, same as RuleBasedPolicy. It
    receives a `_FlatPolicyView` (query, turn, budget, transcript) with no
    C1-C5 component slots, so it structurally cannot reach for candidate
    pools, curated sets, or any other structured harness state.
    """

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        read_doc_ids = {
            step.arguments.get("doc_id")
            for step in state.transcript
            if step.action == "read_document" and step.arguments.get("doc_id")
        }
        searched = any(step.action in ("search_corpus", "grep_corpus") for step in state.transcript)

        if not searched:
            return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})

        unread_doc_ids = []
        for step in state.transcript:
            if step.action in ("search_corpus", "grep_corpus"):
                for item in step.observation.get("results", []):
                    doc_id = item.get("doc_id")
                    if doc_id and doc_id not in read_doc_ids:
                        unread_doc_ids.append(doc_id)
        if unread_doc_ids:
            return HarnessAction.from_dict({"action": "read_document", "arguments": {"doc_id": unread_doc_ids[0]}})

        if state.turn >= 3 or read_doc_ids:
            answer, cited = self._compose_answer_from_transcript(state)
            return HarnessAction.from_dict({
                "action": "end_search",
                "arguments": {"reason": "rule_based_flat_policy_complete", "answer": answer, "cited_chunk_ids": cited},
            })

        return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})

    @staticmethod
    def _compose_answer_from_transcript(state: Any) -> tuple:
        """Produce a minimal answer directly from read_document observations.

        This lives in the test-only RuleBasedFlatPolicy, not in the
        harness: the harness itself performs no synthesis for H0. A real
        model's action policy would generate its own answer text; this
        deterministic stand-in exists only so H0 trajectories are
        testable end-to-end without a real model attached.
        """
        for step in state.transcript:
            if step.action == "read_document":
                results = step.observation.get("results", [])
                if results:
                    chunk = results[0]
                    return chunk.get("content", "")[:500], [chunk.get("chunk_id")]
        return "", []


class TransformersActionPolicy:
    """Backward-compatible H1 policy wrapper over TransformersModelBackend.

    Superseded by ModelBackedEpisodePolicy (agent_policy.py), which is
    usable for all four registered models via model_backend.MODEL_REGISTRY,
    shares a backend across H0/H1/answer-generation roles for the same
    model (Part G), and never silently falls back to a rule-based action
    on malformed output. This class is kept only so existing callers that
    constructed it directly by model_id continue to work; new code should
    use ModelBackedEpisodePolicy/ModelBackedFlatPolicy with a
    TransformersModelBackend from evaluation.py's resolve_*_factory
    functions instead.
    """

    def __init__(self, model_id: str, device_map: str = "auto", max_new_tokens: int = 512):
        from .model_backend import MODEL_REGISTRY as BACKEND_REGISTRY, TransformersModelBackend
        from .agent_policy import ModelBackedEpisodePolicy

        model_name = next((name for name, entry in BACKEND_REGISTRY.items() if entry["model_id"] == model_id), None)
        if model_name is None:
            raise ValueError(
                f"'{model_id}' is not one of the registered model ids in model_backend.MODEL_REGISTRY. "
                f"TransformersActionPolicy only accepts one of the four configured frozen models' exact ids."
            )
        backend = TransformersModelBackend(model_name, device_map=device_map, max_new_tokens=max_new_tokens)
        self._delegate = ModelBackedEpisodePolicy(backend)

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        return self._delegate.choose_action(observation, state)
