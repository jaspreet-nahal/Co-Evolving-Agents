"""Strict H0 executor: a genuinely flat ReAct loop.

This is a separate, minimal execution path -- it does not reuse the
legacy six-stage pipeline, WorkingMemory, SufficiencyChecker, or any
planner-generated fixed sub-query list. The model (via its action policy)
is the only thing deciding whether to search, read, or stop and answer;
the harness performs no implicit multi-query fan-out and no retrieval
deduplication unless the policy itself chooses to avoid repeats.

Loop shape:
    initialize query
    -> model action (search / read / grep / answer)
    -> execute tool (if any) against SearchReadTools with suppress_seen=False
    -> append the raw observation to a growing transcript
    -> model action
    -> ...
    -> model signals end_search, or the hard turn/search budget is reached
    -> the harness calls the shared AnswerGenerator (same model, same
       prompt template as H1) to produce the final answer from whatever
       evidence the transcript actually retrieved
    -> final verifier/evaluation (shared with H1; verification is part of
       evaluation, not part of the harness state intervention under test)

`end_search` is a pure termination signal here (reason only). It does not
carry the final answer -- final-answer generation is a separate,
harness-invoked step so the exact same AnswerGenerator/prompt is used for
both H0 and H1 (see harness.py's _synthesize_from_flat_transcript and
answer_generator.py).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .actions import ActionType, HarnessAction
from .search_read import CorpusIndex, SearchReadTools


# H0 exposes only raw retrieval tools plus the ability to stop and answer.
# Structured H1-only actions (fan_out_search, curate, verify, review_docs)
# are not part of the flat contract: their absence here is what keeps H0
# free of any hidden structured pipeline, not a filter applied elsewhere.
FLAT_REACT_ALLOWED_ACTIONS = {
    ActionType.SEARCH_CORPUS,
    ActionType.GREP_CORPUS,
    ActionType.READ_DOCUMENT,
    ActionType.END_SEARCH,
}


@dataclass
class FlatReActStep:
    turn: int
    action: str
    arguments: Dict[str, Any]
    observation: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class FlatReActExecutorError(RuntimeError):
    pass


class FlatReActExecutor:
    """Runs one H0 trajectory as a flat, model-driven search/read/answer loop.

    No WorkingMemory, no SufficiencyChecker, no CandidatePool/CuratedSet/
    EvidenceGraph/VerificationCache, no fixed planner sub-query list. The
    only persistent state across turns is the raw transcript
    (`self.transcript`) and SearchReadTools' own seen_chunk_ids bookkeeping,
    which is explicitly disabled for suppression purposes
    (`suppress_seen=False`) so retrieval behaves as a stateless tool from
    the harness's perspective, as required for H0.
    """

    def __init__(self, corpus_index: CorpusIndex, policy: Any,
                 max_turns: int = 40, max_search_steps: int = 10,
                 max_chunks_per_search: int = 10):
        self.corpus_index = corpus_index
        self.policy = policy
        self.max_turns = max_turns
        self.max_search_steps = max_search_steps
        self.max_chunks_per_search = max_chunks_per_search
        self.search_tools = SearchReadTools(corpus_index, suppress_seen=False)
        self.transcript: List[FlatReActStep] = []
        self.turn = 0
        self.search_steps_used = 0
        self.terminated = False
        self.termination_reason = ""

    def render_transcript(self, query: str) -> str:
        lines = [f"QUERY: {query}", f"TURN: {self.turn}/{self.max_turns}",
                 f"SEARCH_STEPS_USED: {self.search_steps_used}/{self.max_search_steps}"]
        for step in self.transcript:
            lines.append(f"[turn {step.turn}] ACTION={step.action} ARGS={step.arguments}")
            lines.append(f"[turn {step.turn}] OBSERVATION={step.observation}")
        return "\n".join(lines)

    def _search_budget_remaining(self) -> int:
        return max(0, self.max_search_steps - self.search_steps_used)

    def run(self, query: str) -> "FlatReActExecutor":
        while not self.terminated and self.turn < self.max_turns:
            transcript_text = self.render_transcript(query)
            raw_action = self.policy.choose_action(transcript_text, self._policy_view(query))
            action = raw_action if isinstance(raw_action, HarnessAction) else HarnessAction.from_dict(raw_action)

            if action.action not in FLAT_REACT_ALLOWED_ACTIONS:
                raise FlatReActExecutorError(
                    f"Action '{action.action.value}' is not part of the flat H0 contract. "
                    f"H0 only allows {[a.value for a in FLAT_REACT_ALLOWED_ACTIONS]}."
                )

            observation = self._execute(action)
            self.transcript.append(FlatReActStep(self.turn, action.action.value, action.arguments, observation))
            self.turn += 1

            if action.action == ActionType.END_SEARCH:
                self.terminated = True
                self.termination_reason = str(action.arguments.get("reason", "policy_end"))

        if not self.terminated:
            self.terminated = True
            self.termination_reason = "max_turns"

        return self

    def _policy_view(self, query: str) -> "_FlatPolicyView":
        return _FlatPolicyView(query=query, turn=self.turn, max_turns=self.max_turns,
                                search_steps_used=self.search_steps_used, max_search_steps=self.max_search_steps,
                                transcript=self.transcript)

    def _execute(self, action: HarnessAction) -> Dict[str, Any]:
        args = action.arguments
        if action.action == ActionType.SEARCH_CORPUS:
            if self._search_budget_remaining() <= 0:
                return {"tool": "search_corpus", "results": [], "budget_exhausted": True}
            result = self.search_tools.search_corpus(str(args.get("query", "")), self.max_chunks_per_search)
            self.search_steps_used += 1
            return result
        if action.action == ActionType.GREP_CORPUS:
            if self._search_budget_remaining() <= 0:
                return {"tool": "grep_corpus", "results": [], "budget_exhausted": True}
            result = self.search_tools.grep_corpus(str(args.get("pattern", "")), int(args.get("max_results", 5)))
            self.search_steps_used += 1
            return result
        if action.action == ActionType.READ_DOCUMENT:
            return self.search_tools.read_document(str(args.get("doc_id", "")))
        if action.action == ActionType.END_SEARCH:
            return {"tool": "end_search", "reason": args.get("reason", ""), "answer": args.get("answer", "")}
        raise FlatReActExecutorError(f"Unsupported action: {action.action.value}")


@dataclass
class _FlatPolicyView:
    """Read-only view handed to the H0 policy instead of EpisodeState.

    Deliberately has none of EpisodeState's C1-C5 slots -- a policy cannot
    reach for `state.candidate_pool` here because there is no such
    attribute, not because it happens to be None.
    """
    query: str
    turn: int
    max_turns: int
    search_steps_used: int
    max_search_steps: int
    transcript: List[FlatReActStep]
