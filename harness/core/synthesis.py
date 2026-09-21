from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
import json

from .models import Trajectory, Chunk
from .working_memory import WorkingMemory, EvidenceChunk


@dataclass
class Claim:
    text: str
    citation_chunk_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    claim_type: str = "factual"


@dataclass
class SynthesisResult:
    answer: str
    claims: List[Claim] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)
    reasoning: str = ""
    model_used: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class SynthesisEngine:

    def __init__(self, use_llm: bool = False, llm_client: Any = None,
                 model_name: str = "rule_based"):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.model_name = model_name

    def synthesize(self, query: str, working_memory: WorkingMemory,
                   trajectory: Trajectory, plan_constraints: List = None) -> SynthesisResult:
        if self.use_llm and self.llm_client:
            return self._synthesize_with_llm(query, working_memory, trajectory, plan_constraints)
        else:
            return self._synthesize_rule_based(query, working_memory, trajectory, plan_constraints)

    def _synthesize_rule_based(self, query: str, working_memory: WorkingMemory,
                               trajectory: Trajectory, plan_constraints: List = None) -> SynthesisResult:
        active_chunks = working_memory.get_active_chunks()

        if not active_chunks:
            return SynthesisResult(
                answer="Insufficient evidence to answer the query.",
                claims=[],
                cited_chunk_ids=[],
                reasoning="No evidence chunks available in working memory.",
                model_used=self.model_name
            )

        cited_chunks = []
        claims = []

        if self._is_list_question(query):
            claims = self._extract_list_answers(query, active_chunks)
        else:
            claims = self._extract_factual_answers(query, active_chunks)

        for claim in claims:
            cited_chunks.extend(claim.citation_chunk_ids)

        cited_chunks = list(set(cited_chunks))

        answer = self._format_answer(claims, query)

        reasoning = f"Synthesized from {len(active_chunks)} evidence chunks. " \
                    f"Extracted {len(claims)} claims citing {len(cited_chunks)} unique chunks."

        return SynthesisResult(
            answer=answer,
            claims=claims,
            cited_chunk_ids=cited_chunks,
            reasoning=reasoning,
            model_used=self.model_name
        )

    def _is_list_question(self, query: str) -> bool:
        query_lower = query.lower()
        list_indicators = ['list', 'what are', 'which', 'enumerate', 'all', 'every', 'names of']
        return any(indicator in query_lower for indicator in list_indicators)

    def _extract_list_answers(self, query: str, active_chunks: List[EvidenceChunk]) -> List[Claim]:
        claims = []

        for chunk_evidence in active_chunks:
            chunk = chunk_evidence.chunk
            content = chunk.content

            sentences = re.split(r'[.!?]+', content)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20 and len(sent) < 200:
                    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent):
                        claims.append(Claim(
                            text=sent,
                            citation_chunk_ids=[chunk.chunk_id],
                            confidence=0.6,
                            claim_type="list_item"
                        ))

        return claims[:20]

    def _extract_factual_answers(self, query: str, active_chunks: List[EvidenceChunk]) -> List[Claim]:
        claims = []

        sorted_chunks = sorted(active_chunks, key=lambda x: x.relevance_score, reverse=True)

        for chunk_evidence in sorted_chunks[:5]:
            chunk = chunk_evidence.chunk
            content = chunk.content[:500]

            claims.append(Claim(
                text=content,
                citation_chunk_ids=[chunk.chunk_id],
                confidence=chunk_evidence.relevance_score,
                claim_type="factual"
            ))

        return claims

    def _format_answer(self, claims: List[Claim], query: str) -> str:
        if not claims:
            return "No answer could be synthesized from the available evidence."

        if self._is_list_question(query):
            lines = []
            for i, claim in enumerate(claims, 1):
                citations = ", ".join([f"[{cid}]" for cid in claim.citation_chunk_ids])
                lines.append(f"{i}. {claim.text} {citations}")
            return "\n".join(lines)
        else:
            parts = []
            for claim in claims:
                citations = ", ".join([f"[{cid}]" for cid in claim.citation_chunk_ids])
                parts.append(f"{claim.text} {citations}")
            return " ".join(parts)

    def _synthesize_with_llm(self, query: str, working_memory: WorkingMemory,
                             trajectory: Trajectory, plan_constraints: List = None) -> SynthesisResult:
        return self._synthesize_rule_based(query, working_memory, trajectory, plan_constraints)


def create_synthesis_engine(use_llm: bool = False, llm_client: Any = None,
                            model_name: str = "rule_based") -> SynthesisEngine:
    return SynthesisEngine(use_llm=use_llm, llm_client=llm_client, model_name=model_name)