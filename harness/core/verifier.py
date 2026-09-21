from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from .models import VerificationResult, Trajectory, Chunk
from .working_memory import WorkingMemory, EvidenceChunk
from .search_read import CorpusIndex
from .synthesis import SynthesisResult, Claim


@dataclass
class VerificationConfig:
    check_chunk_existence: bool = True
    check_claim_support: bool = True
    check_citation_format: bool = True
    min_claim_overlap: float = 0.3


class Verifier:

    def __init__(self, corpus_index: CorpusIndex, config: VerificationConfig = None):
        self.corpus_index = corpus_index
        self.config = config or VerificationConfig()

    def verify(self, synthesis_result: SynthesisResult,
               working_memory: WorkingMemory,
               trajectory: Trajectory) -> VerificationResult:
        invalid_citations = []
        claims_verified = []

        for claim in synthesis_result.claims:
            claim_verification = self._verify_claim(claim, working_memory)
            claims_verified.append(claim_verification)

            for citation in claim_verification.get("invalid_citations", []):
                invalid_citations.append(citation)

        all_valid = len(invalid_citations) == 0

        return VerificationResult(
            all_citations_valid=all_valid,
            invalid_citations=invalid_citations,
            claims_verified=claims_verified
        )

    def _verify_claim(self, claim: Claim, working_memory: WorkingMemory) -> Dict[str, Any]:
        result = {
            "claim_text": claim.text,
            "citation_chunk_ids": claim.citation_chunk_ids,
            "valid_citations": [],
            "invalid_citations": [],
            "claim_supported": False,
            "support_details": {}
        }

        for chunk_id in claim.citation_chunk_ids:
            chunk_exists = self._check_chunk_exists(chunk_id, working_memory)

            if not chunk_exists:
                result["invalid_citations"].append({
                    "chunk_id": chunk_id,
                    "reason": "Chunk ID does not exist in corpus or working memory"
                })
                continue

            result["valid_citations"].append(chunk_id)

            if self.config.check_claim_support:
                support = self._check_claim_support(claim.text, chunk_id, working_memory)
                result["support_details"][chunk_id] = support
                if support["supported"]:
                    result["claim_supported"] = True

        return result

    def _check_chunk_exists(self, chunk_id: str, working_memory: WorkingMemory) -> bool:
        if chunk_id in working_memory.evidence_chunks:
            return True

        chunk = self.corpus_index.get_chunk(chunk_id)
        return chunk is not None

    def _check_claim_support(self, claim_text: str, chunk_id: str,
                             working_memory: WorkingMemory) -> Dict[str, Any]:
        chunk = None
        if chunk_id in working_memory.evidence_chunks:
            chunk = working_memory.evidence_chunks[chunk_id].chunk
        else:
            chunk = self.corpus_index.get_chunk(chunk_id)

        if not chunk:
            return {"supported": False, "reason": "Chunk not found", "overlap": 0.0}

        claim_tokens = set(claim_text.lower().split())
        chunk_tokens = set(chunk.content.lower().split())

        if not claim_tokens:
            return {"supported": False, "reason": "Empty claim", "overlap": 0.0}

        overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
        supported = overlap >= self.config.min_claim_overlap

        return {
            "supported": supported,
            "overlap": overlap,
            "threshold": self.config.min_claim_overlap,
            "claim_tokens": len(claim_tokens),
            "chunk_tokens": len(chunk_tokens),
            "shared_tokens": len(claim_tokens & chunk_tokens)
        }

    def verify_trajectory(self, trajectory: Trajectory) -> VerificationResult:
        claims = []
        for chunk_id in trajectory.final_cited_chunk_ids:
            claims.append(Claim(
                text=f"Cited chunk {chunk_id}",
                citation_chunk_ids=[chunk_id]
            ))

        synthesis_result = SynthesisResult(
            answer=trajectory.final_answer,
            claims=claims,
            cited_chunk_ids=trajectory.final_cited_chunk_ids
        )

        working_memory = WorkingMemory()
        return self.verify(synthesis_result, working_memory, trajectory)


def create_verifier(corpus_index: CorpusIndex, config: VerificationConfig = None) -> Verifier:
    return Verifier(corpus_index, config)