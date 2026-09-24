from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from .harness import HarnessConfig, H0_CONFIG_DEFAULTS, H1_CONFIG_DEFAULTS, create_harness
from .model_adapter import RuleBasedPolicy, RuleBasedFlatPolicy
from .model_backend import TransformersModelBackend, MODEL_REGISTRY as MODEL_BACKEND_REGISTRY
from .agent_policy import ModelBackedFlatPolicy, ModelBackedEpisodePolicy
from .answer_generator import RuleBasedAnswerGenerator, ModelBackedAnswerGenerator
from .models import BenchmarkResult


# Test-only entries. The four frozen models are NOT listed with static
# factories here -- their factories are constructed lazily and per-cell by
# _default_policy_factories()/_default_answer_generator_factory() below, so
# that (a) no model weights are touched at import time, and (b) H0 policy,
# H1 policy, and the answer generator for a given model_name all share one
# TransformersModelBackend instance (Part G: literally the same loaded
# model driving every role, not four separately loaded copies).
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rule_based": {"model_id": "rule_based", "status": "test_only",
                    "policy_factory": RuleBasedPolicy, "flat_policy_factory": RuleBasedFlatPolicy},
}


class UnavailableModelError(RuntimeError):
    """Raised when a model has no registered, runnable policy factory.

    For the four frozen research models this no longer means "not in
    MODEL_REGISTRY" (they are always configured, per model_backend.py) --
    it means the backend itself could not be constructed/loaded (auth,
    download, or runtime failure). See model_backend.ModelBackendError and
    its subclasses for the underlying cause. This is never silently
    replaced by RuleBasedPolicy.
    """


def _is_real_model(model_name: str) -> bool:
    return model_name in MODEL_BACKEND_REGISTRY


def _default_backend_factory(model_name: str) -> Callable[[], TransformersModelBackend]:
    """One shared backend instance per (evaluation run, model_name), reused
    across the H0 policy, H1 policy, and answer generator roles for that
    model so they are provably the same loaded model (Part G)."""
    holder: Dict[str, TransformersModelBackend] = {}

    def get_backend() -> TransformersModelBackend:
        if "backend" not in holder:
            holder["backend"] = TransformersModelBackend.from_env(model_name)
        return holder["backend"]

    return get_backend


def resolve_policy_factory(model_name: str, policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
                            backend_factory: Optional[Callable[[], TransformersModelBackend]] = None) -> Callable[[], Any]:
    """Resolve a zero-arg H1 (EpisodeState-facing) policy factory for `model_name`.

    `policy_registry` lets a caller supply/override factories (e.g. a mock
    backend for tests) without editing this module. If omitted and
    `model_name` is one of the four registered frozen models, a
    ModelBackedEpisodePolicy factory bound to `backend_factory` (or a
    fresh TransformersModelBackend) is returned. Any other unregistered
    name raises UnavailableModelError.
    """
    if policy_registry and model_name in policy_registry:
        return policy_registry[model_name]

    if _is_real_model(model_name):
        get_backend = backend_factory or _default_backend_factory(model_name)
        return lambda: ModelBackedEpisodePolicy(get_backend())

    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise UnavailableModelError(
            f"Unknown model '{model_name}'. It is not in model_backend.MODEL_REGISTRY or the "
            f"test-only MODEL_REGISTRY, and no policy_registry override was supplied."
        )
    factory = entry.get("policy_factory")
    if factory is None:
        raise UnavailableModelError(
            f"Model '{model_name}' has status '{entry.get('status')}' and no registered "
            f"policy factory. Refusing to silently fall back to a rule-based policy for an "
            f"actual model experiment. Pass a policy_registry entry for this model to run it."
        )
    return factory


def resolve_flat_policy_factory(model_name: str, flat_policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
                                 backend_factory: Optional[Callable[[], TransformersModelBackend]] = None) -> Callable[[], Any]:
    """Resolve a zero-arg H0 (flat-transcript-facing) policy factory for `model_name`.

    This is a distinct policy shape from the H1 factory
    (`resolve_policy_factory`): FlatReActExecutor hands the policy a
    `_FlatPolicyView` with no C1-C5 component slots, not an EpisodeState.
    For the four frozen models, this resolves to a ModelBackedFlatPolicy
    sharing the same backend as the H1 policy for that model_name.
    """
    if flat_policy_registry and model_name in flat_policy_registry:
        return flat_policy_registry[model_name]

    if _is_real_model(model_name):
        get_backend = backend_factory or _default_backend_factory(model_name)
        return lambda: ModelBackedFlatPolicy(get_backend())

    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise UnavailableModelError(
            f"Unknown model '{model_name}'. It is not in model_backend.MODEL_REGISTRY or the "
            f"test-only MODEL_REGISTRY, and no flat_policy_registry override was supplied."
        )
    factory = entry.get("flat_policy_factory")
    if factory is None:
        raise UnavailableModelError(
            f"Model '{model_name}' has status '{entry.get('status')}' and no registered "
            f"flat policy factory. Refusing to silently fall back to a rule-based policy for "
            f"an actual H0 experiment. Pass a flat_policy_registry entry for this model to run it."
        )
    return factory


def resolve_answer_generator_factory(model_name: str, answer_generator_registry: Optional[Dict[str, Callable[[], Any]]] = None,
                                      backend_factory: Optional[Callable[[], TransformersModelBackend]] = None) -> Callable[[], Any]:
    """Resolve a zero-arg AnswerGenerator factory for `model_name`.

    Real benchmark runs must not receive RuleBasedAnswerGenerator for one
    of the four frozen models; this always returns a
    ModelBackedAnswerGenerator sharing the same backend as that model's
    policies, unless `answer_generator_registry` overrides it (mocks/tests).
    "rule_based" resolves to RuleBasedAnswerGenerator, matching its
    test-only policies.
    """
    if answer_generator_registry and model_name in answer_generator_registry:
        return answer_generator_registry[model_name]

    if _is_real_model(model_name):
        get_backend = backend_factory or _default_backend_factory(model_name)
        return lambda: ModelBackedAnswerGenerator(get_backend())

    if model_name == "rule_based":
        return RuleBasedAnswerGenerator

    raise UnavailableModelError(
        f"Unknown model '{model_name}'. It is not in model_backend.MODEL_REGISTRY or the "
        f"test-only MODEL_REGISTRY, and no answer_generator_registry override was supplied."
    )


@dataclass
class PairedCell:
    model_name: str
    benchmark: str
    off: BenchmarkResult
    on: BenchmarkResult
    score_name: str = "output_recall"
    assistance_gain: float = 0.0
    failure_distribution_delta: Dict[str, int] = field(default_factory=dict)

    def compute(self) -> "PairedCell":
        off_score = getattr(self.off, f"avg_{self.score_name}", 0.0)
        on_score = getattr(self.on, f"avg_{self.score_name}", 0.0)
        self.assistance_gain = on_score - off_score
        off_dist = self.off.metrics.get("failure_mode_distribution", {})
        on_dist = self.on.metrics.get("failure_mode_distribution", {})
        labels = set(off_dist) | set(on_dist)
        self.failure_distribution_delta = {label: on_dist.get(label, 0) - off_dist.get(label, 0) for label in sorted(labels)}
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "score_name": self.score_name,
            "assistance_gain": self.assistance_gain,
            "failure_distribution_delta": self.failure_distribution_delta,
            "off": self.off.to_dict(),
            "on": self.on.to_dict(),
        }


@dataclass
class PairedEvaluation:
    cells: List[PairedCell] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": [cell.to_dict() for cell in self.cells]}


def run_paired_benchmark(
    queries: List[Dict[str, Any]],
    benchmark_name: str,
    corpus_index: Any,
    model_name: str,
    off_config: Optional[HarnessConfig] = None,
    on_config: Optional[HarnessConfig] = None,
    log_dir: str = "harness/logs",
    score_name: str = "output_recall",
    policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
    flat_policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
    answer_generator_registry: Optional[Dict[str, Callable[[], Any]]] = None,
    require_model_backed_answers: Optional[bool] = None,
) -> PairedCell:
    """Run one (model, benchmark) cell as a paired H0/OFF vs H1/ON comparison.

    Never mutates caller-supplied `off_config`/`on_config`: this function
    derives its own per-condition copies via `dataclasses.replace`, so a
    config object the caller reuses across cells is left untouched.

    The canonical component presets from the scientific contract are
    applied here: off_run_config always gets H0_CONFIG_DEFAULTS (00000)
    and on_run_config always gets H1_CONFIG_DEFAULTS (11111), regardless
    of what off_config/on_config happened to carry for the C1-C5 flags.

    For one of the four frozen models (`model_name` in
    model_backend.MODEL_REGISTRY), the H0 policy, H1 policy, and answer
    generator all share one TransformersModelBackend instance, so the same
    loaded model drives every role (Part G fairness rule) --
    `require_model_backed_answers` defaults to True in that case. For
    "rule_based" (or any name overridden via the *_registry parameters),
    it defaults to False, since RuleBasedAnswerGenerator is test-only.
    """
    base_off = off_config or HarnessConfig()
    base_on = on_config or HarnessConfig()

    backend_factory = _default_backend_factory(model_name) if _is_real_model(model_name) else None

    flat_policy_factory = resolve_flat_policy_factory(model_name, flat_policy_registry, backend_factory)
    policy_factory = resolve_policy_factory(model_name, policy_registry, backend_factory)
    answer_generator_factory = resolve_answer_generator_factory(model_name, answer_generator_registry, backend_factory)

    if require_model_backed_answers is None:
        require_model_backed_answers = _is_real_model(model_name) and not answer_generator_registry

    off_run_config = replace(base_off, model_name=model_name, execution_mode="off",
                              action_policy=flat_policy_factory(), answer_generator=answer_generator_factory(),
                              require_model_backed_answers=require_model_backed_answers, **H0_CONFIG_DEFAULTS)
    on_run_config = replace(base_on, model_name=model_name, execution_mode="harness1",
                             action_policy=policy_factory(), answer_generator=answer_generator_factory(),
                             require_model_backed_answers=require_model_backed_answers, **H1_CONFIG_DEFAULTS)

    off_harness = create_harness(corpus_index, off_run_config, log_dir)
    on_harness = create_harness(corpus_index, on_run_config, log_dir)
    off_result = off_harness.run_benchmark([_with_condition(query, "off") for query in queries], benchmark_name)
    on_result = on_harness.run_benchmark([_with_condition(query, "on") for query in queries], benchmark_name)
    return PairedCell(model_name, benchmark_name, off_result, on_result, score_name).compute()


def run_paired_grid(
    query_sets: Dict[str, List[Dict[str, Any]]],
    corpus_factory: Callable[[str], Any],
    models: Optional[List[str]] = None,
    log_dir: str = "harness/logs",
    score_name: str = "output_recall",
    policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
    flat_policy_registry: Optional[Dict[str, Callable[[], Any]]] = None,
    answer_generator_registry: Optional[Dict[str, Callable[[], Any]]] = None,
) -> PairedEvaluation:
    models = models or list(MODEL_BACKEND_REGISTRY)
    evaluation = PairedEvaluation()
    for model_name in models:
        for benchmark_name, queries in query_sets.items():
            evaluation.cells.append(run_paired_benchmark(
                queries=queries,
                benchmark_name=benchmark_name,
                corpus_index=corpus_factory(benchmark_name),
                model_name=model_name,
                log_dir=log_dir,
                score_name=score_name,
                policy_registry=policy_registry,
                flat_policy_registry=flat_policy_registry,
                answer_generator_registry=answer_generator_registry,
            ))
    return evaluation


def _with_condition(query: Dict[str, Any], condition: str) -> Dict[str, Any]:
    """Attach the stable task_id/condition pairing identity to a query dict.

    `task_id` stays identical across the H0/H1 pair (defaults to the
    original query_id) so paired trajectories can be joined reliably.
    `query_id` is still suffixed to keep per-run log directories distinct --
    but query_id suffixing is no longer the only pairing mechanism; task_id
    and condition are stored as first-class Trajectory fields.
    """
    copy = dict(query)
    original_id = copy.get("query_id", "query")
    copy["task_id"] = copy.get("task_id", original_id)
    copy["condition"] = condition
    copy["query_id"] = f"{original_id}_{condition}"
    return copy
