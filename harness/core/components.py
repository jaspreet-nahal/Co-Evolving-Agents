"""Independently switchable H1 state-management components (C1-C5).

Each component owns its own state and its own operations. Nothing outside
a component's own class may create or mutate that component's state. The
orchestrator (harness.py) only calls into a component if the corresponding
HarnessConfig flag is enabled; when disabled, the component object does not
exist at all (its slot on EpisodeState is None), not merely "unused."

This module intentionally does not implement N1-N13 or any component
interaction/ablation logic -- only the five components named in the
scientific contract (C1 Candidate Pool, C2 Curated Evidence Set,
C3 Evidence Graph, C4 Verification Cache, C5 Explicit Sufficiency Check).
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .search_read import CorpusChunk
from .sufficiency_check import SufficiencyChecker, SufficiencyConfig
from .working_memory import WorkingMemory


IMPORTANCE_RANKS = {"very_high": 0, "high": 1, "fair": 2, "low": 3}


@dataclass
class Candidate:
    item_id: str
    doc_id: str
    chunk_ids: List[str] = field(default_factory=list)
    snippet: str = ""
    score: float = 0.0
    source_tool: str = ""
    seen_at_turn: int = 0


class CandidatePool:
    """C1: everything discovered that may be useful later.

    Owns its own dict; nothing else may write to it. A search/read result
    only reaches this pool through `ingest_chunk`, which the orchestrator
    calls only when C1 is enabled.
    """

    def __init__(self):
        self.candidates: Dict[str, Candidate] = {}
        self.duplicate_count: int = 0

    def ingest_chunk(self, chunk: CorpusChunk, source_tool: str, turn: int) -> bool:
        item_id = chunk.doc_id
        candidate = self.candidates.get(item_id)
        if candidate is None:
            self.candidates[item_id] = Candidate(
                item_id=item_id,
                doc_id=chunk.doc_id,
                chunk_ids=[chunk.chunk_id],
                snippet=chunk.content[:240],
                score=chunk.score,
                source_tool=source_tool,
                seen_at_turn=turn,
            )
            return True
        if chunk.chunk_id not in candidate.chunk_ids:
            candidate.chunk_ids.append(chunk.chunk_id)
        candidate.score = max(candidate.score, chunk.score)
        self.duplicate_count += 1
        return False

    def candidate_add(self, chunk: CorpusChunk, source_tool: str, turn: int) -> bool:
        """Explicit model-facing add, alias of ingest_chunk."""
        return self.ingest_chunk(chunk, source_tool, turn)

    def candidate_remove(self, item_id: str) -> bool:
        return self.candidates.pop(item_id, None) is not None

    def candidate_list(self) -> List[str]:
        return list(self.candidates)

    def snapshot(self) -> Dict[str, Any]:
        return {"candidate_ids": list(self.candidates), "duplicate_count": self.duplicate_count}


@dataclass
class CuratedItem:
    item_id: str
    importance: str = "fair"
    rationale: str = ""
    added_at_turn: int = 0


class CuratedSet:
    """C2: the smaller set of evidence selected as worth preserving.

    Depends only on a CandidatePool reference to validate that an item
    being curated actually exists as a candidate -- it never creates
    candidates itself. Auto-seeding (choosing to curate on first search) is
    an orchestration policy applied by the harness only when both C1 and C2
    are enabled; this class exposes curate_add/remove but does not decide
    when to call them on its own.
    """

    def __init__(self, candidate_pool: CandidatePool, max_curated_docs: int = 30):
        self.candidate_pool = candidate_pool
        self.max_curated_docs = max_curated_docs
        self.curated: Dict[str, CuratedItem] = {}

    def curate(self, add_ids: List[str], remove_ids: List[str], importance: Dict[str, str],
               rationale: str = "", turn: int = 0) -> Dict[str, Any]:
        for item_id in remove_ids:
            self.curated.pop(item_id, None)
        added = []
        rejected = []
        for item_id in add_ids:
            if item_id not in self.candidate_pool.candidates:
                continue
            level = importance.get(item_id, "fair")
            if level not in IMPORTANCE_RANKS:
                level = "fair"
            if item_id in self.curated:
                self.curated[item_id].importance = level
                if rationale:
                    self.curated[item_id].rationale = rationale
                continue
            if len(self.curated) >= self.max_curated_docs:
                worst_id = max(self.curated, key=lambda value: IMPORTANCE_RANKS[self.curated[value].importance])
                worst_rank = IMPORTANCE_RANKS[self.curated[worst_id].importance]
                if IMPORTANCE_RANKS[level] >= worst_rank:
                    rejected.append(item_id)
                    continue
                self.curated.pop(worst_id)
            self.curated[item_id] = CuratedItem(item_id, level, rationale, turn)
            added.append(item_id)
        return {"added": added, "removed": remove_ids, "rejected": rejected, "size": len(self.curated)}

    def curate_add(self, item_id: str, importance: str = "fair", rationale: str = "", turn: int = 0) -> Dict[str, Any]:
        return self.curate([item_id], [], {item_id: importance}, rationale, turn)

    def curate_remove(self, item_id: str) -> Dict[str, Any]:
        return self.curate([], [item_id], {})

    def curated_list(self) -> List[str]:
        return self.ordered_curated_ids()

    def ordered_curated_ids(self) -> List[str]:
        return sorted(self.curated, key=lambda item_id: (IMPORTANCE_RANKS[self.curated[item_id].importance], item_id))

    def snapshot(self) -> Dict[str, Any]:
        return {item_id: item.importance for item_id, item in self.curated.items()}


class EvidenceGraph:
    """C3: relationships among documents, entities, claims, and other evidence.

    Owns its own adjacency map. `update` is the ingestion-time write path,
    called by the orchestrator only when C3 is enabled. `graph_query` and
    `graph_neighbors` are the model-facing read operations (Task 3).
    """

    def __init__(self):
        self.edges: Dict[str, Set[str]] = {}

    def update(self, item_id: str, content: str) -> None:
        entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d{4}\b", content))
        for entity in entities:
            self.edges.setdefault(entity, set()).add(item_id)

    def graph_add_node(self, entity: str) -> None:
        self.edges.setdefault(entity, set())

    def graph_add_edge(self, entity: str, item_id: str) -> None:
        self.edges.setdefault(entity, set()).add(item_id)

    def graph_query(self, entity: str) -> List[str]:
        """Return the document ids linked to `entity` (empty if unknown)."""
        return sorted(self.edges.get(entity, set()))

    def graph_neighbors(self, item_id: str) -> List[str]:
        """Return other document ids that share at least one entity with `item_id`."""
        neighbors: Set[str] = set()
        shared_entities = [entity for entity, doc_ids in self.edges.items() if item_id in doc_ids]
        for entity in shared_entities:
            neighbors.update(self.edges[entity])
        neighbors.discard(item_id)
        return sorted(neighbors)

    def snapshot(self) -> Dict[str, Any]:
        return {entity: sorted(doc_ids) for entity, doc_ids in self.edges.items()}


@dataclass
class VerificationRecord:
    claim: str
    item_id: str
    supported: bool
    rationale: str
    created_at_turn: int


class VerificationCache:
    """C4: memoizes previously computed claim/evidence verification results.

    This is purely a cache in front of verification -- it never performs
    verification logic itself (that stays in Verifier /
    _execute_harness1_verify) and verification must remain able to run
    correctly with this component entirely absent (C4 OFF): every check is
    simply recomputed, never cached or reused.
    """

    def __init__(self):
        self.records: Dict[str, VerificationRecord] = {}

    @staticmethod
    def cache_key(claim: str, item_id: str) -> str:
        return f"{item_id}:{claim.strip().lower()}"

    def get(self, claim: str, item_id: str) -> Optional[VerificationRecord]:
        return self.records.get(self.cache_key(claim, item_id))

    def store(self, record: VerificationRecord) -> None:
        self.records[self.cache_key(record.claim, record.item_id)] = record

    def snapshot(self) -> Dict[str, Any]:
        return {"size": len(self.records)}


class SufficiencyDecision(str, Enum):
    SEARCH_AGAIN = "SEARCH_AGAIN"
    ANSWER_NOW = "ANSWER_NOW"


@dataclass
class SufficiencyEvent:
    turn: int
    decision: str
    is_sufficient: bool
    reason: str
    confidence: float


class SufficiencyController:
    """C5: explicit, in-trajectory sufficiency decisions.

    Unlike the legacy post-hoc SufficiencyChecker stage, this component is
    callable mid-trajectory (Task 3): `check_sufficiency` returns a
    structured SEARCH_AGAIN/ANSWER_NOW decision and every call is appended
    to `history` for logging. It wraps the existing rule-based
    SufficiencyChecker logic rather than duplicating it.
    """

    def __init__(self, config: Optional[SufficiencyConfig] = None):
        self.checker = SufficiencyChecker(config)
        self.history: List[SufficiencyEvent] = []

    def check_sufficiency(self, working_memory: WorkingMemory, trajectory, constraints, turn: int) -> SufficiencyEvent:
        result = self.checker.check(working_memory, trajectory, constraints)
        decision = SufficiencyDecision.ANSWER_NOW if result.is_sufficient else SufficiencyDecision.SEARCH_AGAIN
        event = SufficiencyEvent(
            turn=turn,
            decision=decision.value,
            is_sufficient=result.is_sufficient,
            reason=result.reason,
            confidence=result.confidence,
        )
        self.history.append(event)
        return event

    def snapshot(self) -> Dict[str, Any]:
        return {
            "checks": len(self.history),
            "search_again": sum(1 for e in self.history if e.decision == SufficiencyDecision.SEARCH_AGAIN.value),
            "answer_now": sum(1 for e in self.history if e.decision == SufficiencyDecision.ANSWER_NOW.value),
        }
