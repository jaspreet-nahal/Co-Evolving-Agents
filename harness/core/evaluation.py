from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .harness import HarnessConfig, create_harness
from .models import BenchmarkResult


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Muse-Glimmer-30B": {"model_id": "Muse-Glimmer-30B", "status": "label_only"},
    "Qwen3.8-27B": {"model_id": "Qwen3.8-27B", "status": "label_only"},
    "Gemma-4-26B-A4B-it": {"model_id": "google/gemma-4-26B-A4B-it", "status": "label_only"},
    "Gemma-4-31B-it": {"model_id": "google/gemma-4-31B-it", "status": "transformers_policy_supported"},
}


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
) -> PairedCell:
    off_config = off_config or HarnessConfig(model_name=model_name, execution_mode="off")
    on_config = on_config or HarnessConfig(model_name=model_name, execution_mode="harness1")
    off_config.model_name = model_name
    on_config.model_name = model_name
    off_config.execution_mode = "off"
    on_config.execution_mode = "harness1"

    off_harness = create_harness(corpus_index, off_config, log_dir)
    on_harness = create_harness(corpus_index, on_config, log_dir)
    off_result = off_harness.run_benchmark([_with_run_id(query, "off") for query in queries], benchmark_name)
    on_result = on_harness.run_benchmark([_with_run_id(query, "on") for query in queries], benchmark_name)
    return PairedCell(model_name, benchmark_name, off_result, on_result, score_name).compute()


def run_paired_grid(
    query_sets: Dict[str, List[Dict[str, Any]]],
    corpus_factory: Callable[[str], Any],
    models: Optional[List[str]] = None,
    log_dir: str = "harness/logs",
    score_name: str = "output_recall",
) -> PairedEvaluation:
    models = models or list(MODEL_REGISTRY)
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
            ))
    return evaluation


def _with_run_id(query: Dict[str, Any], mode: str) -> Dict[str, Any]:
    copy = dict(query)
    copy["query_id"] = f"{copy.get('query_id', 'query')}_{mode}"
    return copy
