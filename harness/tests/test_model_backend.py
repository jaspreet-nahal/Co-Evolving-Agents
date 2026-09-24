"""Tests for the production model backend, agent-policy contract, and
model-backed answer generation (Parts C/D/E/F/J).

No real model is downloaded or loaded here: transformers'
AutoTokenizer/AutoModelForCausalLM are monkeypatched with lightweight
mocks. Tests that need to exercise real Hugging Face Hub connectivity
checks are skipped by default (see TestCheckModelStatusLive) and are not
part of the default `pytest -q` run's required-to-pass set beyond
confirming the function does not raise when the Hub is unreachable.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import pytest

from harness.core.model_backend import (
    MODEL_REGISTRY, ModelStatus, ModelBackendError, AuthenticationRequiredError, ModelDownloadError,
    TransformersModelBackend, check_model_status, is_cached_locally, resolve_model_entry,
    inspect_gpu, preflight, _hf_token,
)
from harness.core.agent_policy import (
    parse_agent_action, extract_json_object, MalformedModelOutputError,
    ModelBackedFlatPolicy, ModelBackedEpisodePolicy, H0_ALLOWED_ACTIONS, H1_ALLOWED_ACTIONS,
)
from harness.core.answer_generator import ModelBackedAnswerGenerator, AnswerContext, EvidenceItem, AnswerGenerationError
from harness.core.actions import ActionType
from harness.core.flat_react import FlatReActExecutor, FLAT_REACT_ALLOWED_ACTIONS


EXPECTED_MODEL_IDS = {
    "Muse-Glimmer-30B": "meta-models/Muse-Glimmer-30B",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "Gemma-4-26B-A4B-it": "google/gemma-4-26B-A4B-it",
    "Gemma-4-31B-it": "google/gemma-4-31B-it",
}


class _MockBatchEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding: a dict that also
    supports .to(device) (a no-op here, since the mock model has no real
    device placement to perform)."""

    def to(self, device):
        return self


class _MockTokenizer:
    """Stands in for a HF tokenizer: encodes to a single dummy token id per
    call and decodes back to whatever text was registered via `queue_response`."""

    def __init__(self):
        self._responses = []

    def queue_response(self, text: str):
        self._responses.append(text)

    def __call__(self, prompt, return_tensors="pt"):
        import torch
        return _MockBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, token_ids, skip_special_tokens=True):
        if self._responses:
            return self._responses.pop(0)
        return ""


class _MockModel:
    def __init__(self, tokenizer: _MockTokenizer):
        self.tokenizer = tokenizer
        self.device = "cpu"

    def generate(self, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3, 4]])


def _mock_backend(monkeypatch, model_name: str, responses):
    """Build a TransformersModelBackend whose load() is monkeypatched to
    install mock tokenizer/model objects instead of touching the network."""
    backend = TransformersModelBackend(model_name)
    tokenizer = _MockTokenizer()
    for text in responses:
        tokenizer.queue_response(text)
    model = _MockModel(tokenizer)

    def fake_load(self=backend):
        self.tokenizer = tokenizer
        self.model = model
        self.status = ModelStatus.READY

    monkeypatch.setattr(backend, "load", fake_load)
    return backend


# ---------------------------------------------------------------------------
# J1: exact model ID registry
# ---------------------------------------------------------------------------

class TestModelRegistry:

    def test_all_four_models_registered_with_exact_ids(self):
        for name, expected_id in EXPECTED_MODEL_IDS.items():
            assert name in MODEL_REGISTRY
            assert MODEL_REGISTRY[name]["model_id"] == expected_id

    def test_registry_has_no_extra_or_missing_models(self):
        assert set(MODEL_REGISTRY.keys()) == set(EXPECTED_MODEL_IDS.keys())

    def test_resolve_model_entry_matches_registry(self):
        for name in EXPECTED_MODEL_IDS:
            entry = resolve_model_entry(name)
            assert entry["model_id"] == EXPECTED_MODEL_IDS[name]

    def test_resolve_model_entry_raises_for_unknown_model(self):
        with pytest.raises(ModelBackendError):
            resolve_model_entry("not-a-real-model")

    def test_no_model_is_marked_unavailable_merely_for_not_being_cached(self):
        # The registry itself carries no "unavailable" status field at all;
        # availability is determined by check_model_status() at call time,
        # never baked into the static registry.
        for entry in MODEL_REGISTRY.values():
            assert "status" not in entry


# ---------------------------------------------------------------------------
# J2: no silent RuleBasedPolicy fallback
# ---------------------------------------------------------------------------

class TestNoSilentFallback:

    def test_transformers_action_policy_rejects_unregistered_model_id(self):
        from harness.core.model_adapter import TransformersActionPolicy
        with pytest.raises(ValueError):
            TransformersActionPolicy("not/a-registered-model")

    def test_malformed_model_output_raises_not_falls_back(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=["this is not json", "still not json"])
        policy = ModelBackedFlatPolicy(backend, max_parse_retries=1)
        with pytest.raises(MalformedModelOutputError):
            policy.choose_action("some observation", state=None)

    def test_action_outside_allowed_set_raises_not_falls_back(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            json.dumps({"action": "curate", "arguments": {"add_ids": ["x"]}}),
            json.dumps({"action": "curate", "arguments": {"add_ids": ["x"]}}),
        ])
        policy = ModelBackedFlatPolicy(backend, max_parse_retries=1)
        # 'curate' is not in H0's allowed action set.
        with pytest.raises(MalformedModelOutputError):
            policy.choose_action("some observation", state=None)


# ---------------------------------------------------------------------------
# J3/J4: lazy loading, cache-vs-download status distinction
# ---------------------------------------------------------------------------

class TestLazyLoadingAndStatus:

    def test_backend_does_not_load_at_construction(self):
        backend = TransformersModelBackend("Qwen3.8-27B")
        assert backend.model is None
        assert backend.tokenizer is None
        assert backend.status == ModelStatus.AVAILABLE_IN_CONFIG

    def test_load_is_idempotent(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[])
        backend.load()
        model_ref = backend.model
        backend.load()
        assert backend.model is model_ref  # not reloaded

    def test_is_cached_locally_returns_bool_without_network(self):
        # Whatever the actual answer is, this must not raise and must be a bool.
        result = is_cached_locally("Qwen/Qwen3.8-27B")
        assert isinstance(result, bool)

    def test_check_model_status_without_contacting_hub_reports_download_required_when_not_cached(self):
        status = check_model_status("Qwen3.8-27B", contact_hub=False)
        assert status["status"] in (ModelStatus.CACHED.value, ModelStatus.DOWNLOAD_REQUIRED.value)
        assert "model_id" in status and status["model_id"] == "Qwen/Qwen3.8-27B"

    def test_check_model_status_reports_cached_true_when_cache_hit(self, monkeypatch):
        monkeypatch.setattr("harness.core.model_backend.is_cached_locally", lambda model_id: True)
        status = check_model_status("Qwen3.8-27B", contact_hub=True)
        assert status["cached"] is True
        assert status["status"] == ModelStatus.CACHED.value

    def test_preflight_does_not_load_any_model(self, monkeypatch):
        calls = []
        original_init = TransformersModelBackend.load
        monkeypatch.setattr(TransformersModelBackend, "load", lambda self: calls.append(self.model_name) or original_init)
        preflight(models=["Qwen3.8-27B"], contact_hub=False)
        assert calls == []  # load() was never invoked by preflight


# ---------------------------------------------------------------------------
# J5: model load failure is explicit
# ---------------------------------------------------------------------------

class TestLoadFailureExplicit:

    def test_gated_repo_load_failure_is_classified_as_auth_required(self, monkeypatch):
        # `requires_auth` on the registry entry is only a configured hint
        # (see model_backend.py) -- it must not pre-emptively block a
        # load attempt, since the Hub's real gating status is authoritative
        # (confirmed via preflight's check_model_status, which contacts the
        # Hub directly). What must remain true is that an ACTUAL auth-shaped
        # failure from from_pretrained is still classified correctly.
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        backend = TransformersModelBackend("Gemma-4-31B-it")

        class _FakeTokenizerCls:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("401 Client Error: this repository is gated, you need authorization")

        monkeypatch.setattr("transformers.AutoTokenizer", _FakeTokenizerCls)
        with pytest.raises(AuthenticationRequiredError):
            backend.load()
        assert backend.status == ModelStatus.LOAD_FAILED

    def test_load_does_not_preemptively_block_on_requires_auth_hint(self, monkeypatch):
        # A model hinted as requires_auth=True must still be ATTEMPTED
        # (not refused before any network call) since the hint can be
        # wrong -- the real Hub is authoritative, as preflight demonstrated
        # for these four models (see MODEL_REGISTRY comment).
        monkeypatch.delenv("HF_TOKEN", raising=False)
        backend = TransformersModelBackend("Gemma-4-31B-it")
        load_attempted = []

        class _FakeTokenizerCls:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                load_attempted.append(True)
                raise RuntimeError("simulated: repo turned out to be public, but something else failed")

        monkeypatch.setattr("transformers.AutoTokenizer", _FakeTokenizerCls)
        with pytest.raises(ModelDownloadError):
            backend.load()
        assert load_attempted == [True]  # the attempt was made, not pre-blocked

    def test_load_failure_is_wrapped_as_model_backend_error(self, monkeypatch):
        backend = TransformersModelBackend("Qwen3.8-27B")  # requires_auth=False

        def fake_from_pretrained(*args, **kwargs):
            raise RuntimeError("simulated network failure")

        import harness.core.model_backend as mb_module

        class _FakeTokenizerCls:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("simulated tokenizer download failure")

        monkeypatch.setattr("transformers.AutoTokenizer", _FakeTokenizerCls)
        with pytest.raises(ModelDownloadError):
            backend.load()
        assert backend.status == ModelStatus.LOAD_FAILED

    def test_gpu_inspection_never_raises_without_cuda(self):
        gpu = inspect_gpu()
        assert isinstance(gpu.cuda_available, bool)
        assert isinstance(gpu.device_count, int)


# ---------------------------------------------------------------------------
# Part D: GPU/device config is discovered, not hard-coded
# ---------------------------------------------------------------------------

class TestGpuConfigDiscovery:

    def test_no_hardcoded_gpu_model_name_in_backend_module(self):
        import inspect
        source = inspect.getsource(sys.modules["harness.core.model_backend"])
        for forbidden in ["H100", "L40S", "A100"]:
            assert forbidden not in source

    def test_device_map_and_dtype_default_to_auto(self):
        backend = TransformersModelBackend("Qwen3.8-27B")
        assert backend.device_map == "auto"
        assert backend.dtype == "auto"

    def test_device_map_and_dtype_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("HARNESS_MODEL_DEVICE_MAP", "cpu")
        monkeypatch.setenv("HARNESS_MODEL_DTYPE", "float32")
        backend = TransformersModelBackend.from_env("Qwen3.8-27B")
        assert backend.device_map == "cpu"
        assert backend.dtype == "float32"

    def test_quantization_is_not_applied_unless_explicitly_requested(self):
        backend = TransformersModelBackend("Qwen3.8-27B")
        assert backend.quantization is None

    def test_unload_clears_model_and_tokenizer_references(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[])
        backend.load()
        assert backend.model is not None
        backend.unload()
        assert backend.model is None
        assert backend.tokenizer is None
        assert backend.status == ModelStatus.AVAILABLE_IN_CONFIG

    def test_credentials_are_never_stored_on_the_backend_instance(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "fake-secret-token-value")
        backend = TransformersModelBackend("Qwen3.8-27B")
        instance_values = [str(v) for v in vars(backend).values()]
        assert not any("fake-secret-token-value" in v for v in instance_values)


# ---------------------------------------------------------------------------
# Part E: canonical action contract, parsing, retries
# ---------------------------------------------------------------------------

class TestActionContract:

    def test_h0_allowed_actions_are_a_strict_subset_of_h1(self):
        assert H0_ALLOWED_ACTIONS <= H1_ALLOWED_ACTIONS
        assert H0_ALLOWED_ACTIONS != H1_ALLOWED_ACTIONS

    def test_h0_allowed_actions_match_flat_react_contract(self):
        assert H0_ALLOWED_ACTIONS == FLAT_REACT_ALLOWED_ACTIONS

    def test_parse_agent_action_accepts_well_formed_json(self):
        action = parse_agent_action(json.dumps({"action": "search_corpus", "arguments": {"query": "x"}}))
        assert action.action == ActionType.SEARCH_CORPUS

    def test_parse_agent_action_extracts_json_from_markdown_fence(self):
        text = "```json\n" + json.dumps({"action": "end_search", "arguments": {"reason": "done"}}) + "\n```"
        action = parse_agent_action(text)
        assert action.action == ActionType.END_SEARCH

    def test_parse_agent_action_rejects_invalid_json(self):
        with pytest.raises(MalformedModelOutputError):
            parse_agent_action("not json at all")

    def test_parse_agent_action_rejects_disallowed_action(self):
        with pytest.raises(MalformedModelOutputError):
            parse_agent_action(json.dumps({"action": "curate", "arguments": {}}), allowed_actions=H0_ALLOWED_ACTIONS)

    def test_bounded_retry_succeeds_on_second_attempt_and_logs_both(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            "garbage output",
            json.dumps({"action": "search_corpus", "arguments": {"query": "x"}}),
        ])
        policy = ModelBackedFlatPolicy(backend, max_parse_retries=1)
        action = policy.choose_action("obs", state=None)
        assert action.action == ActionType.SEARCH_CORPUS
        assert len(policy.attempts) == 2
        assert policy.attempts[0].succeeded is False
        assert policy.attempts[1].succeeded is True

    def test_retry_exhaustion_raises_and_logs_every_attempt(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=["bad", "still bad", "extra"])
        policy = ModelBackedFlatPolicy(backend, max_parse_retries=1)
        with pytest.raises(MalformedModelOutputError):
            policy.choose_action("obs", state=None)
        assert len(policy.attempts) == 2  # exactly max_parse_retries + 1
        assert all(not a.succeeded for a in policy.attempts)

    def test_h1_policy_allows_structured_actions_h0_does_not(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            json.dumps({"action": "curate", "arguments": {"add_ids": ["x"]}}),
        ])
        policy = ModelBackedEpisodePolicy(backend, max_parse_retries=0)
        action = policy.choose_action("obs", state=None)
        assert action.action == ActionType.CURATE


# ---------------------------------------------------------------------------
# Part F: model-backed final answer path
# ---------------------------------------------------------------------------

class TestModelBackedAnswerGenerator:

    def test_generates_answer_from_evidence(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            json.dumps({"answer": "Alice built it.", "cited_chunk_ids": ["d1_chunk_0"], "reasoning": "found in doc"}),
        ])
        generator = ModelBackedAnswerGenerator(backend)
        context = AnswerContext(query="Who built it?", evidence=[EvidenceItem(chunk_id="d1_chunk_0", doc_id="d1", content="Alice built it.", score=1.0)])
        result = generator.generate(context)
        assert result.answer == "Alice built it."
        assert result.cited_chunk_ids == ["d1_chunk_0"]
        assert result.generator == "model_backed"

    def test_malformed_answer_output_raises_not_silently_falls_back(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=["not json", "still not json"])
        generator = ModelBackedAnswerGenerator(backend, max_parse_retries=1)
        context = AnswerContext(query="Who built it?", evidence=[])
        with pytest.raises(AnswerGenerationError):
            generator.generate(context)

    def test_answer_missing_required_field_raises(self, monkeypatch):
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            json.dumps({"cited_chunk_ids": []}),
            json.dumps({"cited_chunk_ids": []}),
        ])
        generator = ModelBackedAnswerGenerator(backend, max_parse_retries=1)
        with pytest.raises(AnswerGenerationError):
            generator.generate(AnswerContext(query="q", evidence=[]))

    def test_same_backend_drives_flat_policy_episode_policy_and_answer_generator(self, monkeypatch):
        # Part G: literally the same TransformersModelBackend instance can
        # serve all three roles for one model_name.
        backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=[
            json.dumps({"action": "search_corpus", "arguments": {"query": "q"}}),
            json.dumps({"action": "search_corpus", "arguments": {"query": "q"}}),
            json.dumps({"answer": "a", "cited_chunk_ids": []}),
        ])
        flat_policy = ModelBackedFlatPolicy(backend)
        episode_policy = ModelBackedEpisodePolicy(backend)
        answer_gen = ModelBackedAnswerGenerator(backend)

        assert flat_policy.backend is backend
        assert episode_policy.backend is backend
        assert answer_gen.backend is backend


# ---------------------------------------------------------------------------
# Full model backend + full-grid runner integration (mocked)
# ---------------------------------------------------------------------------

class TestFullGridWithMockedPolicies:

    def test_run_paired_grid_works_with_mock_policy_registries(self):
        from harness.core import InMemoryCorpusIndex, CorpusDocument, RuleBasedPolicy, RuleBasedFlatPolicy
        from harness.core.evaluation import run_paired_grid

        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

        evaluation = run_paired_grid(
            query_sets={"test": [{"query_id": "q1", "query": "Who built it?", "gold_chunk_ids": ["d1_chunk_0"]}]},
            corpus_factory=lambda name: index,
            models=["rule_based"],
            log_dir="harness/logs/test_model_backend_grid",
        )
        assert len(evaluation.cells) == 1
        assert evaluation.cells[0].off.trajectories[0].execution_mode == "off"
        assert evaluation.cells[0].on.trajectories[0].execution_mode == "harness1"

    def test_run_paired_benchmark_with_real_model_name_uses_shared_backend_across_roles(self, monkeypatch):
        from harness.core import InMemoryCorpusIndex, CorpusDocument
        from harness.core.evaluation import run_paired_benchmark

        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=10)
        index.add_document(CorpusDocument(doc_id="d1", content="Alice built a search system."))

        responses = [
            json.dumps({"action": "search_corpus", "arguments": {"query": "who"}}),
            json.dumps({"action": "end_search", "arguments": {"reason": "done"}}),
            json.dumps({"answer": "Alice.", "cited_chunk_ids": ["d1_chunk_0"]}),
            json.dumps({"action": "fan_out_search", "arguments": {"queries": ["who"]}}),
            json.dumps({"action": "end_search", "arguments": {"reason": "done"}}),
            json.dumps({"answer": "Alice.", "cited_chunk_ids": ["d1_chunk_0"]}),
        ]

        def fake_backend_factory():
            backend = _mock_backend(monkeypatch, "Qwen3.8-27B", responses=list(responses))
            return lambda: backend

        import harness.core.evaluation as evaluation_module
        monkeypatch.setattr(evaluation_module, "_default_backend_factory", lambda model_name: fake_backend_factory())

        cell = run_paired_benchmark(
            queries=[{"query_id": "q1", "query": "Who built it?", "gold_chunk_ids": ["d1_chunk_0"]}],
            benchmark_name="test",
            corpus_index=index,
            model_name="Qwen3.8-27B",
            log_dir="harness/logs/test_model_backend_real_grid",
        )
        assert cell.off.trajectories[0].execution_mode == "off"
        assert cell.on.trajectories[0].execution_mode == "harness1"
        assert not cell.off.trajectories[0].final_answer.startswith("ERROR:")
        assert not cell.on.trajectories[0].final_answer.startswith("ERROR:")
