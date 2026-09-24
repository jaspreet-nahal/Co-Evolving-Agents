"""Final-answer generation, decoupled from action-selection.

Both H0 and H1 call the SAME AnswerGenerator instance (same model, same
system prompt, same generation settings) after their respective research
loops complete. The harness supplies the evidence context (raw retrieved
chunks for H0, working-memory/curated chunks for H1) but never rewrites or
post-processes the model's answer text itself -- what the model returns is
the trajectory's final_answer, verbatim.

RuleBasedAnswerGenerator preserves the old extractive-template behavior
and exists only as a test fallback (see HarnessConfig.require_model_backed_answers,
which real benchmark runs must set True).
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent_policy import extract_json_object
from .model_backend import TransformersModelBackend


@dataclass
class EvidenceItem:
    chunk_id: str
    doc_id: str
    content: str
    score: float = 0.0


@dataclass
class AnswerContext:
    query: str
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str
    cited_chunk_ids: List[str] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    generator: str = ""


class AnswerGenerationError(RuntimeError):
    """Raised when a model-backed answer generator cannot produce or parse
    a final answer. Never caught to silently substitute a rule-based
    answer in real-mode runs."""


class AnswerGenerator:
    def generate(self, context: AnswerContext) -> AnswerResult:
        raise NotImplementedError


class RuleBasedAnswerGenerator(AnswerGenerator):
    """Deterministic extractive answer generator. Test-only: see
    HarnessConfig.require_model_backed_answers, which real benchmark runs
    must set so this class is refused at trajectory-run time.
    """

    def generate(self, context: AnswerContext) -> AnswerResult:
        if not context.evidence:
            return AnswerResult(
                answer="Insufficient evidence to answer the query.",
                cited_chunk_ids=[],
                claims=[],
                reasoning="No evidence available.",
                generator="rule_based",
            )

        is_list_question = any(w in context.query.lower() for w in ["list", "what are", "which", "enumerate", "all", "every", "names of"])
        sorted_evidence = sorted(context.evidence, key=lambda item: item.score, reverse=True)

        if is_list_question:
            claims = []
            for item in sorted_evidence:
                for sentence in re.split(r"[.!?]+", item.content):
                    sentence = sentence.strip()
                    if 20 < len(sentence) < 200 and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence):
                        claims.append({"text": sentence, "citations": [item.chunk_id], "confidence": 0.6, "type": "list_item"})
            claims = claims[:20]
        else:
            claims = [
                {"text": item.content[:500], "citations": [item.chunk_id], "confidence": item.score, "type": "factual"}
                for item in sorted_evidence[:5]
            ]

        if not claims:
            return AnswerResult(
                answer="No answer could be synthesized from the available evidence.",
                cited_chunk_ids=[], claims=[], reasoning="No claims extracted.", generator="rule_based",
            )

        cited = list(dict.fromkeys(cid for claim in claims for cid in claim["citations"]))
        if is_list_question:
            lines = [f"{i}. {claim['text']} " + ", ".join(f"[{cid}]" for cid in claim["citations"])
                     for i, claim in enumerate(claims, 1)]
            answer = "\n".join(lines)
        else:
            parts = [f"{claim['text']} " + ", ".join(f"[{cid}]" for cid in claim["citations"]) for claim in claims]
            answer = " ".join(parts)

        return AnswerResult(answer=answer, cited_chunk_ids=cited, claims=claims,
                             reasoning=f"Synthesized from {len(context.evidence)} evidence items.", generator="rule_based")


class ModelBackedAnswerGenerator(AnswerGenerator):
    """Produces the final answer using the same frozen model as the action
    policy, via a single generation call.

    Output contract (kept distinct from the action contract, since this is
    a different kind of model turn): a JSON object
        {"answer": "...", "cited_chunk_ids": ["..."], "reasoning": "..."}
    Malformed output raises AnswerGenerationError -- it is never silently
    replaced by RuleBasedAnswerGenerator or by naive text truncation.

    The exact same system prompt template and generation settings
    (`backend.max_new_tokens`, greedy decoding) are used regardless of
    whether the caller is H0 or H1; only `context.evidence` differs, which
    is the harness-supplied research result, not a stronger prompt.
    """

    system_prompt = (
        "You are answering a research question using only the evidence provided below. "
        "Cite the chunk_id of every piece of evidence you rely on. "
        "Respond with exactly one JSON object and nothing else: "
        "{\"answer\": \"...\", \"cited_chunk_ids\": [\"...\"], \"reasoning\": \"...\"}."
    )

    def __init__(self, backend: TransformersModelBackend, max_parse_retries: int = 1):
        self.backend = backend
        self.max_parse_retries = max_parse_retries

    def _render_evidence(self, context: AnswerContext) -> str:
        if not context.evidence:
            return "EVIDENCE: none retrieved."
        lines = ["EVIDENCE:"]
        for item in context.evidence:
            lines.append(f"[chunk_id={item.chunk_id} doc_id={item.doc_id} score={item.score:.4f}]\n{item.content}")
        return "\n".join(lines)

    def generate(self, context: AnswerContext) -> AnswerResult:
        prompt = f"{self.system_prompt}\nQUERY: {context.query}\n{self._render_evidence(context)}\n"

        last_error: Optional[Exception] = None
        current_prompt = prompt
        for _ in range(self.max_parse_retries + 1):
            raw_text = self.backend.generate(current_prompt)
            try:
                return self._parse(raw_text)
            except AnswerGenerationError as error:
                last_error = error
                current_prompt = (
                    f"{prompt}\n\nYour previous response could not be parsed: {error}\n"
                    f"Respond again with exactly one valid JSON object as instructed."
                )
        raise last_error

    def _parse(self, raw_text: str) -> AnswerResult:
        candidate = extract_json_object(raw_text)
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise AnswerGenerationError(f"Model answer output is not valid JSON: {error}. Raw text: {raw_text!r}") from error

        if not isinstance(value, dict) or "answer" not in value:
            raise AnswerGenerationError(f"Model answer output is missing required 'answer' field. Raw text: {raw_text!r}")

        cited = value.get("cited_chunk_ids", [])
        if not isinstance(cited, list):
            raise AnswerGenerationError(f"Model answer output's cited_chunk_ids must be a list. Raw text: {raw_text!r}")

        return AnswerResult(
            answer=str(value["answer"]),
            cited_chunk_ids=[str(c) for c in cited],
            claims=[{"text": str(value["answer"]), "citations": [str(c) for c in cited], "confidence": 1.0, "type": "model_answer"}],
            reasoning=str(value.get("reasoning", "")),
            generator="model_backed",
        )
