import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .search_read import CorpusChunk, CorpusDocument


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


@dataclass
class CuratedItem:
    item_id: str
    importance: str = "fair"
    rationale: str = ""
    added_at_turn: int = 0


@dataclass
class VerificationRecord:
    claim: str
    item_id: str
    supported: bool
    rationale: str
    created_at_turn: int


@dataclass
class ActionRecord:
    turn: int
    action: str
    arguments: Dict[str, Any]
    result_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EpisodeState:
    query: str
    max_turns: int = 40
    max_curated_docs: int = 30
    context_budget_chars: int = 30000
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    curated: Dict[str, CuratedItem] = field(default_factory=dict)
    document_store: Dict[str, CorpusDocument] = field(default_factory=dict)
    evidence_graph: Dict[str, Set[str]] = field(default_factory=dict)
    verification_cache: Dict[str, VerificationRecord] = field(default_factory=dict)
    action_history: List[ActionRecord] = field(default_factory=list)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    seen_chunk_ids: Set[str] = field(default_factory=set)
    turn: int = 0
    terminated: bool = False
    termination_reason: str = ""
    auto_seeded: bool = False
    duplicate_count: int = 0

    def record_action(self, action: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.action_history.append(ActionRecord(self.turn, action, arguments, result))

    def add_document(self, document: CorpusDocument) -> None:
        self.document_store[document.doc_id] = document
        for chunk in document.chunks:
            self.ingest_chunk(chunk, "read_document")

    def ingest_chunk(self, chunk: CorpusChunk, source_tool: str, turn: Optional[int] = None) -> bool:
        self.seen_chunk_ids.add(chunk.chunk_id)
        item_id = chunk.doc_id
        candidate = self.candidates.get(item_id)
        if candidate is None:
            candidate = Candidate(
                item_id=item_id,
                doc_id=chunk.doc_id,
                chunk_ids=[chunk.chunk_id],
                snippet=chunk.content[:240],
                score=chunk.score,
                source_tool=source_tool,
                seen_at_turn=self.turn if turn is None else turn,
            )
            self.candidates[item_id] = candidate
            self._update_graph(item_id, chunk.content)
            return True
        if chunk.chunk_id not in candidate.chunk_ids:
            candidate.chunk_ids.append(chunk.chunk_id)
        candidate.score = max(candidate.score, chunk.score)
        self.duplicate_count += 1
        return False

    def _update_graph(self, item_id: str, content: str) -> None:
        entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d{4}\b", content))
        for entity in entities:
            self.evidence_graph.setdefault(entity, set()).add(item_id)

    def curate(self, add_ids: List[str], remove_ids: List[str], importance: Dict[str, str], rationale: str = "") -> Dict[str, Any]:
        for item_id in remove_ids:
            self.curated.pop(item_id, None)
        added = []
        rejected = []
        for item_id in add_ids:
            if item_id not in self.candidates:
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
            self.curated[item_id] = CuratedItem(item_id, level, rationale, self.turn)
            added.append(item_id)
        return {"added": added, "removed": remove_ids, "rejected": rejected, "size": len(self.curated)}

    def ordered_curated_ids(self) -> List[str]:
        return sorted(self.curated, key=lambda item_id: (IMPORTANCE_RANKS[self.curated[item_id].importance], item_id))

    def cache_key(self, claim: str, item_id: str) -> str:
        return f"{item_id}:{claim.strip().lower()}"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "turn": self.turn,
            "candidate_ids": list(self.candidates),
            "curated": {item_id: item.importance for item_id, item in self.curated.items()},
            "document_ids": list(self.document_store),
            "evidence_graph": {entity: sorted(doc_ids) for entity, doc_ids in self.evidence_graph.items()},
            "verification_cache_size": len(self.verification_cache),
            "search_count": len(self.search_history),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "duplicate_count": self.duplicate_count,
        }