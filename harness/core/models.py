from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json


class StageName(str, Enum):
    PLANNER = "planner"
    SEARCH_READ = "search_read"
    WORKING_MEMORY = "working_memory"
    SUFFICIENCY_CHECK = "sufficiency_check"
    SYNTHESIS = "synthesis"
    VERIFIER = "verifier"


@dataclass
class Constraint:
    type: str
    description: str
    raw_text: str
    parsed_value: Any = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class SearchResult:
    chunks: List[Chunk]
    query: str
    tool_used: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StageLog:
    stage: StageName
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


@dataclass
class Trajectory:
    query_id: str
    query: str
    model_name: str
    benchmark: str
    task_id: str = ""
    trial_id: str = ""
    condition: str = ""
    budget: Dict[str, Any] = field(default_factory=dict)
    component_fingerprint: Dict[str, bool] = field(default_factory=dict)
    sufficiency_events: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    stage_logs: List[StageLog] = field(default_factory=list)
    all_retrieved_chunk_ids: List[str] = field(default_factory=list)
    final_cited_chunk_ids: List[str] = field(default_factory=list)
    gold_relevant_chunk_ids: List[str] = field(default_factory=list)
    final_answer: str = ""
    trajectory_recall: float = 0.0
    output_recall: float = 0.0
    sufficiency_decision: bool = False
    sufficiency_reason: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_mode: str = "legacy"
    termination_reason: str = ""
    curated_document_ids: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    curated_set_recall: float = 0.0
    claim_coverage: float = 0.0
    citation_support: float = 0.0
    citation_accuracy: float = 0.0
    mandatory_requirement_compliance: Optional[bool] = None
    turns: int = 0
    search_calls: int = 0
    read_calls: int = 0
    unique_sources: int = 0
    repeated_actions: int = 0
    search_branching: int = 0
    backtracking: int = 0
    time_to_first_evidence_ms: Optional[float] = None
    time_to_sufficiency_ms: Optional[float] = None
    total_duration_ms: float = 0.0
    failure_mode: str = ""
    primary_failure_category: str = ""
    secondary_failure_categories: List[str] = field(default_factory=list)
    first_failure_turn: Optional[int] = None
    recovery_turn: Optional[int] = None
    final_failure_category: str = ""

    def add_stage_log(self, log: StageLog):
        self.stage_logs.append(log)

    def compute_recalls(self):
        if not self.gold_relevant_chunk_ids:
            self.trajectory_recall = 0.0
            self.output_recall = 0.0
            return

        gold_set = set(self.gold_relevant_chunk_ids)
        retrieved_set = set(self.all_retrieved_chunk_ids)
        cited_set = set(self.final_cited_chunk_ids)

        self.trajectory_recall = len(gold_set & retrieved_set) / len(gold_set)
        self.output_recall = len(gold_set & cited_set) / len(gold_set)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "condition": self.condition,
            "budget": self.budget,
            "component_fingerprint": self.component_fingerprint,
            "sufficiency_events": self.sufficiency_events,
            "constraints": [{"type": c.type, "description": c.description, "raw_text": c.raw_text, "parsed_value": c.parsed_value} for c in self.constraints],
            "stage_logs": [log.to_dict() for log in self.stage_logs],
            "all_retrieved_chunk_ids": self.all_retrieved_chunk_ids,
            "final_cited_chunk_ids": self.final_cited_chunk_ids,
            "gold_relevant_chunk_ids": self.gold_relevant_chunk_ids,
            "final_answer": self.final_answer,
            "trajectory_recall": self.trajectory_recall,
            "output_recall": self.output_recall,
            "sufficiency_decision": self.sufficiency_decision,
            "sufficiency_reason": self.sufficiency_reason,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
            ,"execution_mode": self.execution_mode,
            "termination_reason": self.termination_reason,
            "curated_document_ids": self.curated_document_ids,
            "action_history": self.action_history
            ,"claims": self.claims
            ,"verification": self.verification
            ,"curated_set_recall": self.curated_set_recall
            ,"claim_coverage": self.claim_coverage
            ,"citation_support": self.citation_support
            ,"citation_accuracy": self.citation_accuracy
            ,"mandatory_requirement_compliance": self.mandatory_requirement_compliance
            ,"turns": self.turns
            ,"search_calls": self.search_calls
            ,"read_calls": self.read_calls
            ,"unique_sources": self.unique_sources
            ,"repeated_actions": self.repeated_actions
            ,"search_branching": self.search_branching
            ,"backtracking": self.backtracking
            ,"time_to_first_evidence_ms": self.time_to_first_evidence_ms
            ,"time_to_sufficiency_ms": self.time_to_sufficiency_ms
            ,"total_duration_ms": self.total_duration_ms
            ,"failure_mode": self.failure_mode
            ,"primary_failure_category": self.primary_failure_category
            ,"secondary_failure_categories": self.secondary_failure_categories
            ,"first_failure_turn": self.first_failure_turn
            ,"recovery_turn": self.recovery_turn
            ,"final_failure_category": self.final_failure_category
        }


@dataclass
class SufficiencyResult:
    is_sufficient: bool
    reason: str
    missing_info: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VerificationResult:
    all_citations_valid: bool
    invalid_citations: List[Dict[str, Any]] = field(default_factory=list)
    claims_verified: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    benchmark: str
    model_name: str
    trajectories: List[Trajectory] = field(default_factory=list)
    avg_trajectory_recall: float = 0.0
    avg_output_recall: float = 0.0
    sufficiency_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    total_queries: int = 0
    completed_queries: int = 0
    failed_queries: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def compute_aggregates(self):
        if not self.trajectories:
            return
        self.total_queries = len(self.trajectories)
        self.completed_queries = sum(1 for t in self.trajectories if t.completed_at)
        self.failed_queries = self.total_queries - self.completed_queries
        self.avg_trajectory_recall = sum(t.trajectory_recall for t in self.trajectories) / self.total_queries
        self.avg_output_recall = sum(t.output_recall for t in self.trajectories) / self.total_queries
        self.sufficiency_accuracy = sum(t.sufficiency_decision for t in self.trajectories) / self.total_queries
        self.citation_accuracy = sum(t.citation_accuracy for t in self.trajectories) / self.total_queries
        self.metrics = {
            "avg_curated_set_recall": sum(t.curated_set_recall for t in self.trajectories) / self.total_queries,
            "avg_claim_coverage": sum(t.claim_coverage for t in self.trajectories) / self.total_queries,
            "avg_citation_support": sum(t.citation_support for t in self.trajectories) / self.total_queries,
            "avg_turns": sum(t.turns for t in self.trajectories) / self.total_queries,
            "avg_search_calls": sum(t.search_calls for t in self.trajectories) / self.total_queries,
            "avg_read_calls": sum(t.read_calls for t in self.trajectories) / self.total_queries,
            "avg_unique_sources": sum(t.unique_sources for t in self.trajectories) / self.total_queries,
            "avg_total_duration_ms": sum(t.total_duration_ms for t in self.trajectories) / self.total_queries,
            "failure_mode_distribution": self._failure_distribution(),
        }

    def _failure_distribution(self) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for trajectory in self.trajectories:
            distribution[trajectory.failure_mode] = distribution.get(trajectory.failure_mode, 0) + 1
        return distribution

    def to_dict(self) -> Dict:
        return {
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "avg_trajectory_recall": self.avg_trajectory_recall,
            "avg_output_recall": self.avg_output_recall,
            "sufficiency_accuracy": self.sufficiency_accuracy,
            "citation_accuracy": self.citation_accuracy,
            "total_queries": self.total_queries,
            "completed_queries": self.completed_queries,
            "failed_queries": self.failed_queries,
            "metrics": self.metrics,
            "trajectories": [t.to_dict() for t in self.trajectories]
        }