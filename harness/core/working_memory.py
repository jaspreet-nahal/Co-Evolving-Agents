from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .models import Chunk, CorpusChunk
from .search_read import CorpusChunk as SearchReadCorpusChunk


@dataclass
class EvidenceChunk:
    chunk: CorpusChunk
    source_tool: str
    retrieval_step: int
    query_or_pattern: str
    added_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    is_pruned: bool = False
    pruned_at: Optional[datetime] = None
    prune_reason: str = ""


class WorkingMemory:

    def __init__(self):
        self.evidence_chunks: Dict[str, EvidenceChunk] = {}
        self.retrieval_history: List[Dict[str, Any]] = []
        self.step_counter = 0

    def add_chunks(self, chunks: List[CorpusChunk], source_tool: str,
                   query_or_pattern: str, relevance_scores: List[float] = None):
        self.step_counter += 1
        relevance_scores = relevance_scores or [0.0] * len(chunks)

        added = []
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id not in self.evidence_chunks:
                evidence = EvidenceChunk(
                    chunk=chunk,
                    source_tool=source_tool,
                    retrieval_step=self.step_counter,
                    query_or_pattern=query_or_pattern,
                    relevance_score=relevance_scores[i] if i < len(relevance_scores) else 0.0
                )
                self.evidence_chunks[chunk.chunk_id] = evidence
                added.append(chunk.chunk_id)

        self.retrieval_history.append({
            "step": self.step_counter,
            "tool": source_tool,
            "query_or_pattern": query_or_pattern,
            "num_chunks_retrieved": len(chunks),
            "num_chunks_added": len(added),
            "added_chunk_ids": added,
            "timestamp": datetime.now().isoformat()
        })

        return added

    def prune_chunks(self, chunk_ids: List[str], reason: str = "") -> List[str]:
        pruned = []
        for chunk_id in chunk_ids:
            if chunk_id in self.evidence_chunks and not self.evidence_chunks[chunk_id].is_pruned:
                self.evidence_chunks[chunk_id].is_pruned = True
                self.evidence_chunks[chunk_id].pruned_at = datetime.now()
                self.evidence_chunks[chunk_id].prune_reason = reason
                pruned.append(chunk_id)
        return pruned

    def get_active_chunks(self) -> List[EvidenceChunk]:
        return [ec for ec in self.evidence_chunks.values() if not ec.is_pruned]

    def get_all_chunks(self) -> List[EvidenceChunk]:
        return list(self.evidence_chunks.values())

    def get_chunk_ids(self, include_pruned: bool = False) -> List[str]:
        if include_pruned:
            return list(self.evidence_chunks.keys())
        return [cid for cid, ec in self.evidence_chunks.items() if not ec.is_pruned]

    def get_context_for_synthesis(self, max_chunks: int = 50, max_chars_per_chunk: int = 1000) -> str:
        active = self.get_active_chunks()

        active.sort(key=lambda x: x.relevance_score, reverse=True)

        active = active[:max_chunks]

        context_parts = []
        for ec in active:
            chunk = ec.chunk
            content = chunk.content[:max_chars_per_chunk]
            if len(chunk.content) > max_chars_per_chunk:
                content += "..."

            context_parts.append(
                f"[CHUNK_ID: {chunk.chunk_id} | DOC_ID: {chunk.doc_id} | "
                f"SOURCE: {ec.source_tool} | QUERY: {ec.query_or_pattern}]\n"
                f"{content}\n"
            )

        return "\n---\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.evidence_chunks)
        pruned = sum(1 for ec in self.evidence_chunks.values() if ec.is_pruned)
        active = total - pruned

        return {
            "total_chunks": total,
            "active_chunks": active,
            "pruned_chunks": pruned,
            "retrieval_steps": self.step_counter,
            "retrieval_history": self.retrieval_history
        }

    def reset(self):
        self.evidence_chunks = {}
        self.retrieval_history = []
        self.step_counter = 0