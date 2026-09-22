from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .models import Trajectory, Chunk


@dataclass
class RecallMetrics:
    trajectory_recall: float = 0.0
    output_recall: float = 0.0
    gold_chunks_total: int = 0
    gold_chunks_encountered: int = 0
    gold_chunks_cited: int = 0
    encountered_chunk_ids: List[str] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)
    gold_chunk_ids: List[str] = field(default_factory=list)
    failure_mode: str = ""


class MetricsCalculator:

    @staticmethod
    def enrich_trajectory(trajectory: Trajectory) -> None:
        trajectory.compute_recalls()
        trajectory.turns = len(trajectory.action_history) or len(trajectory.stage_logs)
        actions = trajectory.action_history
        search_names = {"fan_out_search", "search_corpus", "grep_corpus"}
        trajectory.search_calls = sum(item.get("action") in search_names for item in actions)
        trajectory.read_calls = sum(item.get("action") == "read_document" for item in actions)
        trajectory.repeated_actions = max(0, len(actions) - len({(item.get("action"), str(item.get("arguments", {}))) for item in actions}))
        trajectory.search_branching = sum(
            max(0, len(item.get("arguments", {}).get("queries", [])) - 1)
            for item in actions if item.get("action") == "fan_out_search"
        )
        trajectory.backtracking = sum(
            1 for item in actions
            if item.get("action") in {"curate", "verify"} and item.get("turn", 0) > 0
        )
        trajectory.unique_sources = len({item.split("_chunk_", 1)[0] for item in trajectory.all_retrieved_chunk_ids})
        elapsed_ms = 0.0
        for log in trajectory.stage_logs:
            elapsed_ms += log.duration_ms
            if log.stage.value == "search_read" and trajectory.all_retrieved_chunk_ids:
                trajectory.time_to_first_evidence_ms = elapsed_ms
            if log.stage.value == "sufficiency_check":
                trajectory.time_to_sufficiency_ms = elapsed_ms
        curated_ids = set(trajectory.curated_document_ids)
        gold_docs = {item.split("_chunk_", 1)[0] for item in trajectory.gold_relevant_chunk_ids}
        trajectory.curated_set_recall = len(curated_ids & gold_docs) / len(gold_docs) if gold_docs else 0.0

        synthesis_log = next((log for log in reversed(trajectory.stage_logs) if log.stage.value == "synthesis"), None)
        verifier_log = next((log for log in reversed(trajectory.stage_logs) if log.stage.value == "verifier"), None)
        if synthesis_log:
            trajectory.claims = synthesis_log.output_data.get("claims", [])
        if verifier_log:
            trajectory.verification = verifier_log.output_data
            verified = verifier_log.output_data.get("claims_verified", [])
            supported = [item for item in verified if item.get("claim_supported")]
            trajectory.citation_support = len(supported) / len(verified) if verified else 0.0
            trajectory.citation_accuracy = 1.0 if verifier_log.output_data.get("all_citations_valid") else 0.0
            trajectory.claim_coverage = len(verified) / len(trajectory.claims) if trajectory.claims else 0.0

        trajectory.failure_mode = MetricsCalculator._determine_failure_mode(
            RecallMetrics(
                trajectory_recall=trajectory.trajectory_recall,
                output_recall=trajectory.output_recall,
                gold_chunks_total=len(trajectory.gold_relevant_chunk_ids),
            )
        )
        trajectory.primary_failure_category = trajectory.failure_mode
        if trajectory.failure_mode == "success":
            trajectory.final_failure_category = "none"
        else:
            trajectory.final_failure_category = trajectory.failure_mode

    @staticmethod
    def calculate(trajectory: Trajectory) -> RecallMetrics:
        gold_set = set(trajectory.gold_relevant_chunk_ids)
        retrieved_set = set(trajectory.all_retrieved_chunk_ids)
        cited_set = set(trajectory.final_cited_chunk_ids)

        metrics = RecallMetrics()
        metrics.gold_chunks_total = len(gold_set)
        metrics.gold_chunk_ids = list(gold_set)
        metrics.encountered_chunk_ids = list(retrieved_set)
        metrics.cited_chunk_ids = list(cited_set)

        if metrics.gold_chunks_total == 0:
            metrics.trajectory_recall = 0.0
            metrics.output_recall = 0.0
            metrics.failure_mode = "no_gold_standard"
            return metrics

        encountered_gold = gold_set & retrieved_set
        metrics.gold_chunks_encountered = len(encountered_gold)
        metrics.trajectory_recall = metrics.gold_chunks_encountered / metrics.gold_chunks_total

        cited_gold = gold_set & cited_set
        metrics.gold_chunks_cited = len(cited_gold)
        metrics.output_recall = metrics.gold_chunks_cited / metrics.gold_chunks_total

        metrics.failure_mode = MetricsCalculator._determine_failure_mode(metrics)

        return metrics

    @staticmethod
    def _determine_failure_mode(metrics: RecallMetrics) -> str:
        if metrics.trajectory_recall == 1.0 and metrics.output_recall == 1.0:
            return "success"
        elif metrics.trajectory_recall > 0.5 and metrics.output_recall < metrics.trajectory_recall:
            return "found_not_used"
        elif metrics.trajectory_recall < 0.5:
            return "never_found"
        elif metrics.trajectory_recall > 0 and metrics.output_recall == 0:
            return "found_not_used"
        else:
            return "partial"

    @staticmethod
    def aggregate_metrics(trajectories: List[Trajectory]) -> Dict[str, Any]:
        if not trajectories:
            return {}

        all_metrics = [MetricsCalculator.calculate(t) for t in trajectories]

        return {
            "num_trajectories": len(trajectories),
            "avg_trajectory_recall": sum(m.trajectory_recall for m in all_metrics) / len(all_metrics),
            "avg_output_recall": sum(m.output_recall for m in all_metrics) / len(all_metrics),
            "failure_mode_distribution": MetricsCalculator._failure_mode_distribution(all_metrics),
            "trajectory_recall_std": MetricsCalculator._std_dev([m.trajectory_recall for m in all_metrics]),
            "output_recall_std": MetricsCalculator._std_dev([m.output_recall for m in all_metrics]),
            "per_trajectory": [
                {
                    "query_id": t.query_id,
                    "trajectory_recall": m.trajectory_recall,
                    "output_recall": m.output_recall,
                    "failure_mode": m.failure_mode,
                    "gold_chunks_total": m.gold_chunks_total,
                    "gold_chunks_encountered": m.gold_chunks_encountered,
                    "gold_chunks_cited": m.gold_chunks_cited
                }
                for t, m in zip(trajectories, all_metrics)
            ]
        }

    @staticmethod
    def _failure_mode_distribution(metrics_list: List[RecallMetrics]) -> Dict[str, int]:
        dist = {}
        for m in metrics_list:
            dist[m.failure_mode] = dist.get(m.failure_mode, 0) + 1
        return dist

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5


def calculate_recall_metrics(trajectory: Trajectory) -> RecallMetrics:
    return MetricsCalculator.calculate(trajectory)


def aggregate_recall_metrics(trajectories: List[Trajectory]) -> Dict[str, Any]:
    return MetricsCalculator.aggregate_metrics(trajectories)