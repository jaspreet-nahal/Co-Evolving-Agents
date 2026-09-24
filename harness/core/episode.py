from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .search_read import CorpusChunk, CorpusDocument
from .components import CandidatePool, CuratedSet, EvidenceGraph, VerificationCache, SufficiencyController, VerificationRecord


@dataclass
class ActionRecord:
    turn: int
    action: str
    arguments: Dict[str, Any]
    result_summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ComponentUnavailableError(RuntimeError):
    """Raised when code tries to use a C1-C5 component that is OFF for this episode.

    This is intentional and load-bearing for the scientific contract: a
    disabled component must not exist, not merely go unused. If something
    reaches for `state.candidate_pool` while C1 is OFF, that is a bug in
    the caller, not a case to silently no-op.
    """


class EpisodeState:
    """Per-query Harness-1 episode state.

    Each of C1 (candidate_pool), C2 (curated_set), C3 (evidence_graph),
    C4 (verification_cache), and C5 (sufficiency) is an independent,
    optional component. When a component's enable flag is False, its slot
    on this object is None -- the component is not merely idle, it does
    not exist. Code that needs a component must go through the accessor
    (`require_candidate_pool()` etc.), which raises ComponentUnavailableError
    rather than silently creating or no-op'ing the component.

    Document memory (`document_store`) and raw retrieval dedup
    (`seen_chunk_ids`) are not part of C1-C5; they are baseline retrieval
    bookkeeping available in H1 regardless of which of C1-C5 are enabled
    (H1's tools are the same tools as H0's, just with structured state
    layered on top per the enabled components).
    """

    def __init__(self, query: str, max_turns: int = 40, max_curated_docs: int = 30,
                 context_budget_chars: int = 30000,
                 enable_candidate_pool: bool = True,
                 enable_curated_set: bool = True,
                 enable_evidence_graph: bool = True,
                 enable_verification_cache: bool = True,
                 enable_sufficiency_check: bool = True,
                 sufficiency_config: Any = None):
        self.query = query
        self.max_turns = max_turns
        self.max_curated_docs = max_curated_docs
        self.context_budget_chars = context_budget_chars

        self.enable_candidate_pool = enable_candidate_pool
        self.enable_curated_set = enable_curated_set
        self.enable_evidence_graph = enable_evidence_graph
        self.enable_verification_cache = enable_verification_cache
        self.enable_sufficiency_check = enable_sufficiency_check

        self.candidate_pool: Optional[CandidatePool] = CandidatePool() if enable_candidate_pool else None
        self.curated_set: Optional[CuratedSet] = (
            CuratedSet(self.candidate_pool, max_curated_docs)
            if enable_curated_set and self.candidate_pool is not None
            else None
        )
        self.evidence_graph: Optional[EvidenceGraph] = EvidenceGraph() if enable_evidence_graph else None
        self.verification_cache: Optional[VerificationCache] = VerificationCache() if enable_verification_cache else None
        self.sufficiency: Optional[SufficiencyController] = (
            SufficiencyController(sufficiency_config) if enable_sufficiency_check else None
        )

        self.document_store: Dict[str, CorpusDocument] = {}
        self.seen_chunk_ids: Set[str] = set()
        self.action_history: List[ActionRecord] = []
        self.search_history: List[Dict[str, Any]] = []
        self.turn: int = 0
        self.terminated: bool = False
        self.termination_reason: str = ""
        self.auto_seeded: bool = False

    # -- component accessors: raise instead of silently creating/no-oping --

    def require_candidate_pool(self) -> CandidatePool:
        if self.candidate_pool is None:
            raise ComponentUnavailableError("C1 Candidate Pool is disabled for this episode.")
        return self.candidate_pool

    def require_curated_set(self) -> CuratedSet:
        if self.curated_set is None:
            raise ComponentUnavailableError("C2 Curated Evidence Set is disabled for this episode.")
        return self.curated_set

    def require_evidence_graph(self) -> EvidenceGraph:
        if self.evidence_graph is None:
            raise ComponentUnavailableError("C3 Evidence Graph is disabled for this episode.")
        return self.evidence_graph

    def require_verification_cache(self) -> VerificationCache:
        if self.verification_cache is None:
            raise ComponentUnavailableError("C4 Verification Cache is disabled for this episode.")
        return self.verification_cache

    def require_sufficiency(self) -> SufficiencyController:
        if self.sufficiency is None:
            raise ComponentUnavailableError("C5 Explicit Sufficiency Check is disabled for this episode.")
        return self.sufficiency

    # -- baseline retrieval bookkeeping (not a C1-C5 component) --

    def record_action(self, action: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.action_history.append(ActionRecord(self.turn, action, arguments, result))

    def add_document(self, document: CorpusDocument) -> None:
        self.document_store[document.doc_id] = document
        for chunk in document.chunks:
            self.ingest_chunk(chunk, "read_document")

    def ingest_chunk(self, chunk: CorpusChunk, source_tool: str) -> bool:
        """Record a retrieved chunk as seen, and feed it to C1/C3 if enabled.

        This never implicitly enables a component: if C1 is OFF, no
        candidate is created; if C3 is OFF, the graph is not updated. The
        return value reports whether this was the first time this
        document was seen (used for turn bookkeeping), independent of
        whether C1 exists to record it as a "candidate."
        """
        is_new_chunk = chunk.chunk_id not in self.seen_chunk_ids
        self.seen_chunk_ids.add(chunk.chunk_id)

        is_new_candidate = False
        if self.candidate_pool is not None:
            is_new_candidate = self.candidate_pool.ingest_chunk(chunk, source_tool, self.turn)
            if is_new_candidate and self.evidence_graph is not None:
                self.evidence_graph.update(chunk.doc_id, chunk.content)
            return is_new_candidate

        return is_new_chunk

    def curate(self, add_ids: List[str], remove_ids: List[str], importance: Dict[str, str], rationale: str = "") -> Dict[str, Any]:
        return self.require_curated_set().curate(add_ids, remove_ids, importance, rationale, self.turn)

    def ordered_curated_ids(self) -> List[str]:
        if self.curated_set is None:
            return []
        return self.curated_set.ordered_curated_ids()

    def cache_key(self, claim: str, item_id: str) -> str:
        return VerificationCache.cache_key(claim, item_id)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "turn": self.turn,
            "enabled_components": {
                "candidate_pool": self.enable_candidate_pool,
                "curated_set": self.enable_curated_set,
                "evidence_graph": self.enable_evidence_graph,
                "verification_cache": self.enable_verification_cache,
                "sufficiency_check": self.enable_sufficiency_check,
            },
            "candidate_pool": self.candidate_pool.snapshot() if self.candidate_pool else None,
            "curated_set": self.curated_set.snapshot() if self.curated_set else None,
            "document_ids": list(self.document_store),
            "evidence_graph": self.evidence_graph.snapshot() if self.evidence_graph else None,
            "verification_cache": self.verification_cache.snapshot() if self.verification_cache else None,
            "sufficiency": self.sufficiency.snapshot() if self.sufficiency else None,
            "search_count": len(self.search_history),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }
