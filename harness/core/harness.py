import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from .models import (Trajectory, Constraint, StageLog, StageName, BenchmarkResult,SearchResult, SufficiencyResult, VerificationResult)
from .logger import StageLogger, ConsoleLogger
from .planner import Planner, Plan, Constraint as PlannerConstraint
from .search_read import SearchReadTools, CorpusIndex, InMemoryCorpusIndex
from .working_memory import WorkingMemory, EvidenceChunk
from .sufficiency_check import SufficiencyChecker, SufficiencyConfig, SufficiencyCriterion
from .synthesis import SynthesisEngine, SynthesisResult
from .verifier import Verifier, VerificationConfig
from .metrics import calculate_recall_metrics, RecallMetrics, MetricsCalculator
from .actions import ActionType, HarnessAction
from .episode import EpisodeState, ComponentUnavailableError
from .components import VerificationRecord, SufficiencyDecision
from .model_adapter import RuleBasedPolicy, RuleBasedFlatPolicy
from .observations import ObservationRenderer
from .flat_react import FlatReActExecutor
from .answer_generator import AnswerGenerator, AnswerContext, EvidenceItem, RuleBasedAnswerGenerator, AnswerGenerationError


@dataclass
class HarnessConfig:
    max_search_steps: int = 10
    max_chunks_per_search: int = 10
    sufficiency_config: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    verifier_config: VerificationConfig = field(default_factory=VerificationConfig)
    use_llm_planner: bool = False
    use_llm_synthesis: bool = False
    llm_client: Any = None
    model_name: str = "rule_based"
    execution_mode: str = "legacy"
    max_turns: int = 40
    max_curated_docs: int = 30
    context_budget_chars: int = 30000
    auto_seed: bool = True
    action_policy: Any = None

    # The final-answer generator shared by both H0 and H1 (Part F/G). If
    # None, a RuleBasedAnswerGenerator is used -- but that is allowed only
    # when require_model_backed_answers is False (the default, for tests).
    # Real benchmark runs must set require_model_backed_answers=True and
    # supply a model-backed answer_generator; run_trajectory refuses to
    # proceed with a rule-based generator in that mode rather than silently
    # downgrading the experiment.
    answer_generator: Any = None
    require_model_backed_answers: bool = False

    # C1-C5: independently configurable H1 components. These are the only
    # switches the scientific contract defines. The baseline uses two fixed
    # presets -- H0_CONFIG_DEFAULTS (00000) and H1_CONFIG_DEFAULTS (11111)
    # -- but any of the 32 combinations is a valid HarnessConfig and can be
    # executed through run_harness1_episode; only 00000 and 11111 are used
    # in the first-phase experiment.
    enable_candidate_pool: bool = True
    enable_curated_set: bool = True
    enable_evidence_graph: bool = True
    enable_verification_cache: bool = True
    enable_sufficiency_check: bool = True

    def budget_snapshot(self) -> Dict[str, Any]:
        """Canonical, loggable view of the hard experiment budget.

        This is the single source of truth read by both the strict H0
        (FlatReActExecutor) and Harness-1 (H1) execution loops, so paired
        runs are guaranteed to share the exact same budget configuration.
        """
        return {
            "max_turns": self.max_turns,
            "max_search_steps": self.max_search_steps,
            "max_chunks_per_search": self.max_chunks_per_search,
        }

    def component_fingerprint(self) -> Dict[str, bool]:
        """The C1-C5 on/off fingerprint for this config, for logging."""
        return {
            "candidate_pool": self.enable_candidate_pool,
            "curated_set": self.enable_curated_set,
            "evidence_graph": self.enable_evidence_graph,
            "verification_cache": self.enable_verification_cache,
            "sufficiency_check": self.enable_sufficiency_check,
        }


# Canonical baseline presets named in the scientific contract. H0 disables
# all five components (00000); H1 enables all five (11111). Component-level
# ablations (10000, 01000, ...) are supported by HarnessConfig itself but
# are not part of the first-phase baseline run policy.
H0_CONFIG_DEFAULTS: Dict[str, bool] = {
    "enable_candidate_pool": False,
    "enable_curated_set": False,
    "enable_evidence_graph": False,
    "enable_verification_cache": False,
    "enable_sufficiency_check": False,
}

H1_CONFIG_DEFAULTS: Dict[str, bool] = {
    "enable_candidate_pool": True,
    "enable_curated_set": True,
    "enable_evidence_graph": True,
    "enable_verification_cache": True,
    "enable_sufficiency_check": True,
}

class DeepResearchHarness:

    def __init__(self, corpus_index: CorpusIndex, config: HarnessConfig = None,
                 log_dir: str = "harness/logs"):
        self.corpus_index = corpus_index
        self.config = config or HarnessConfig()
        self.logger = StageLogger(log_dir)
        self.console = ConsoleLogger()

        self.planner = Planner(use_llm=self.config.use_llm_planner, llm_client=self.config.llm_client)
        self.search_tools = SearchReadTools(corpus_index)
        self.working_memory = WorkingMemory()
        self.sufficiency_checker = SufficiencyChecker(self.config.sufficiency_config)
        self.synthesis_engine = SynthesisEngine(
            use_llm=self.config.use_llm_synthesis,
            llm_client=self.config.llm_client,
            model_name=self.config.model_name
        )
        self.verifier = Verifier(corpus_index, self.config.verifier_config)
        self.answer_generator: AnswerGenerator = self.config.answer_generator or RuleBasedAnswerGenerator()
        if self.config.require_model_backed_answers and isinstance(self.answer_generator, RuleBasedAnswerGenerator):
            raise AnswerGenerationError(
                "HarnessConfig.require_model_backed_answers is True but no model-backed "
                "answer_generator was supplied; refusing to run a real-mode trajectory with "
                "a rule-based final-answer generator. Pass a ModelBackedAnswerGenerator."
            )

        self.current_trajectory: Optional[Trajectory] = None
        self.plan: Optional[Plan] = None

    def run_trajectory(self, query: str, query_id: str = None,
                       gold_chunk_ids: List[str] = None,
                       benchmark: str = "unknown",
                       task_id: str = None,
                       trial_id: str = "0",
                       condition: str = None) -> Trajectory:
        if self.config.execution_mode == "harness1":
            return self.run_harness1_episode(query, query_id, gold_chunk_ids, benchmark,
                                              task_id=task_id, trial_id=trial_id, condition=condition)
        if self.config.execution_mode == "off":
            return self.run_flat_h0_trajectory(query, query_id, gold_chunk_ids, benchmark,
                                                task_id=task_id, trial_id=trial_id, condition=condition)
        return self.run_legacy_trajectory(query, query_id, gold_chunk_ids, benchmark,
                                           task_id=task_id, trial_id=trial_id, condition=condition)

    def run_legacy_trajectory(self, query: str, query_id: str = None,
                       gold_chunk_ids: List[str] = None,
                       benchmark: str = "unknown",
                       task_id: str = None,
                       trial_id: str = "0",
                       condition: str = None) -> Trajectory:
        """The original fixed six-stage pipeline.

        Kept for backward compatibility and existing tests. This is NOT the
        strict H0 executor -- it retains WorkingMemory and an unconditional
        SufficiencyChecker, which are exactly the kind of hidden structured
        state H0 must not have. `execution_mode="off"` no longer routes
        here; use `execution_mode="legacy"` explicitly if this pipeline is
        needed for its own sake.
        """
        query_id = query_id or str(uuid.uuid4())[:8]
        gold_chunk_ids = gold_chunk_ids or []

        self.current_trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
            task_id=task_id or query_id,
            trial_id=trial_id,
            condition=condition or self.config.execution_mode,
            budget=self.config.budget_snapshot(),
            component_fingerprint=H0_CONFIG_DEFAULTS.copy(),
            gold_relevant_chunk_ids=gold_chunk_ids,
            execution_mode=self.config.execution_mode,
        )
        self.logger.start_trajectory(query_id)

        self.search_tools.reset()
        self.working_memory.reset()

        self.console.info(f"Starting trajectory {query_id}: {query[:100]}...")

        try:
            self._run_planner_stage()
            self._run_search_read_loop()
            self._run_working_memory_stage()
            self._run_sufficiency_check_stage()
            self._run_synthesis_stage()
            self._run_verifier_stage()
            self._compute_final_metrics()

        except Exception as e:
            self.console.error(f"Trajectory {query_id} failed: {e}")
            self.current_trajectory.final_answer = f"ERROR: {str(e)}"

        self.current_trajectory.completed_at = datetime.now()
        self._finalize_trajectory(self.current_trajectory)
        self.logger.save_trajectory_summary(self.current_trajectory)
        return self.current_trajectory

    def run_flat_h0_trajectory(self, query: str, query_id: str = None,
                       gold_chunk_ids: List[str] = None,
                       benchmark: str = "unknown",
                       task_id: str = None,
                       trial_id: str = "0",
                       condition: str = None) -> Trajectory:
        """Strict H0: a genuinely flat, model-driven ReAct loop.

        Uses FlatReActExecutor exclusively. No WorkingMemory, no
        SufficiencyChecker, no CandidatePool/CuratedSet/EvidenceGraph/
        VerificationCache, no planner-generated fixed sub-query list, and
        SearchReadTools runs with suppress_seen=False so retrieval behaves
        as a stateless tool. The verifier is shared with H1 because
        verification is part of evaluation, not the harness state
        intervention under test.
        """
        query_id = query_id or str(uuid.uuid4())[:8]
        gold_chunk_ids = gold_chunk_ids or []

        trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
            task_id=task_id or query_id,
            trial_id=trial_id,
            condition=condition or "off",
            budget=self.config.budget_snapshot(),
            component_fingerprint=H0_CONFIG_DEFAULTS.copy(),
            gold_relevant_chunk_ids=gold_chunk_ids,
            execution_mode="off",
        )
        self.current_trajectory = trajectory
        self.logger.start_trajectory(query_id)
        self.console.info(f"Starting flat H0 trajectory {query_id}: {query[:100]}...")

        policy = self.config.action_policy or RuleBasedFlatPolicy()
        executor = FlatReActExecutor(
            self.corpus_index, policy,
            max_turns=self.config.max_turns,
            max_search_steps=self.config.max_search_steps,
            max_chunks_per_search=self.config.max_chunks_per_search,
        )

        try:
            executor.run(query)

            trajectory.all_retrieved_chunk_ids = executor.search_tools.get_seen_chunk_ids()
            trajectory.termination_reason = executor.termination_reason
            trajectory.action_history = [
                {"turn": step.turn, "action": step.action, "arguments": step.arguments, "result": step.observation}
                for step in executor.transcript
            ]
            trajectory.search_calls = executor.search_steps_used

            trajectory.add_stage_log(StageLog(
                stage=StageName.SEARCH_READ,
                input_data={"mode": "off", "query": query, "budget": trajectory.budget},
                output_data={
                    "actions": trajectory.action_history,
                    "search_steps_used": executor.search_steps_used,
                    "max_search_steps": self.config.max_search_steps,
                    "termination_reason": executor.termination_reason,
                },
            ))

            self._synthesize_from_flat_transcript(trajectory, executor)
            self._verify_flat_trajectory(trajectory, executor)
            self._compute_final_metrics()
        except Exception as error:
            self.console.error(f"Flat H0 trajectory {query_id} failed: {error}")
            trajectory.final_answer = f"ERROR: {error}"

        trajectory.completed_at = datetime.now()
        self._finalize_trajectory(trajectory)
        self.logger.save_trajectory_summary(trajectory)
        return trajectory

    def _synthesize_from_flat_transcript(self, trajectory: Trajectory, executor: FlatReActExecutor) -> None:
        """Generate the final answer via the shared AnswerGenerator (Part F/G).

        H0 has no WorkingMemory or SynthesisEngine stage. The evidence
        context handed to the answer generator is built purely from what
        the flat transcript actually retrieved (search_corpus/grep_corpus/
        read_document observations) -- the harness constructs this context
        but never writes the answer text itself; that is the model's
        (or RuleBasedAnswerGenerator's, in test mode) job. The exact same
        AnswerGenerator instance and prompt template used for H1 is used
        here, per the fairness rule (Part G).
        """
        context = AnswerContext(query=trajectory.query, evidence=self._evidence_from_flat_transcript(executor))
        start_time = time.time()
        result = self.answer_generator.generate(context)
        duration = (time.time() - start_time) * 1000

        trajectory.final_answer = result.answer
        trajectory.final_cited_chunk_ids = result.cited_chunk_ids
        trajectory.claims = result.claims

        trajectory.add_stage_log(StageLog(
            stage=StageName.SYNTHESIS,
            input_data={"mode": "off", "generator": type(self.answer_generator).__name__, "evidence_count": len(context.evidence)},
            output_data={"answer": result.answer, "cited_chunk_ids": result.cited_chunk_ids, "claims": result.claims, "reasoning": result.reasoning},
            duration_ms=duration,
        ))

    @staticmethod
    def _evidence_from_flat_transcript(executor: FlatReActExecutor) -> List[EvidenceItem]:
        seen_chunk_ids = set()
        evidence: List[EvidenceItem] = []
        for step in executor.transcript:
            if step.action in ("search_corpus", "grep_corpus"):
                for item in step.observation.get("results", []):
                    chunk_id = item.get("chunk_id")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        evidence.append(EvidenceItem(chunk_id=chunk_id, doc_id=item.get("doc_id", ""),
                                                      content=item.get("content", ""), score=item.get("score", 0.0)))
            elif step.action == "read_document":
                for item in step.observation.get("results", []):
                    chunk_id = item.get("chunk_id")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        evidence.append(EvidenceItem(chunk_id=chunk_id, doc_id=item.get("doc_id", ""), content=item.get("content", ""), score=0.0))
        return evidence

    def _verify_flat_trajectory(self, trajectory: Trajectory, executor: FlatReActExecutor) -> None:
        from .synthesis import Claim, SynthesisResult
        claims = [Claim(text=c["text"], citation_chunk_ids=c["citations"], confidence=c["confidence"], claim_type=c["type"])
                  for c in trajectory.claims]
        synthesis_result = SynthesisResult(answer=trajectory.final_answer, claims=claims, cited_chunk_ids=trajectory.final_cited_chunk_ids)

        empty_working_memory = WorkingMemory()
        verification_result = self.verifier.verify(synthesis_result, empty_working_memory, trajectory)

        output_data = {
            "all_citations_valid": verification_result.all_citations_valid,
            "invalid_citations": verification_result.invalid_citations,
            "claims_verified": verification_result.claims_verified,
        }
        trajectory.verification = output_data
        trajectory.add_stage_log(StageLog(stage=StageName.VERIFIER, input_data={"mode": "off"}, output_data=output_data))

    def run_harness1_episode(self, query: str, query_id: str = None,
                             gold_chunk_ids: List[str] = None,
                             benchmark: str = "unknown",
                             task_id: str = None,
                             trial_id: str = "0",
                             condition: str = None) -> Trajectory:
        query_id = query_id or str(uuid.uuid4())[:8]
        trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
            task_id=task_id or query_id,
            trial_id=trial_id,
            condition=condition or "harness1",
            budget=self.config.budget_snapshot(),
            component_fingerprint=self.config.component_fingerprint(),
            gold_relevant_chunk_ids=gold_chunk_ids or [],
            execution_mode="harness1",
        )
        self.current_trajectory = trajectory
        self.logger.start_trajectory(query_id)
        self.search_tools.reset()
        self.working_memory.reset()
        self.plan = self.planner.plan(query)
        trajectory.constraints = [
            Constraint(type=item.type, description=item.description, raw_text=item.raw_text, parsed_value=item.parsed_value)
            for item in self.plan.constraints
        ]
        trajectory.add_stage_log(StageLog(
            stage=StageName.PLANNER,
            input_data={"query": query, "mode": "harness1"},
            output_data={"constraints": [item.description for item in trajectory.constraints], "sub_queries": [item.text for item in self.plan.sub_queries]},
        ))
        state = EpisodeState(
            query=query,
            max_turns=self.config.max_turns,
            max_curated_docs=self.config.max_curated_docs,
            context_budget_chars=self.config.context_budget_chars,
            enable_candidate_pool=self.config.enable_candidate_pool,
            enable_curated_set=self.config.enable_curated_set,
            enable_evidence_graph=self.config.enable_evidence_graph,
            enable_verification_cache=self.config.enable_verification_cache,
            enable_sufficiency_check=self.config.enable_sufficiency_check,
            sufficiency_config=self.config.sufficiency_config,
        )
        policy = self.config.action_policy or RuleBasedPolicy()
        renderer = ObservationRenderer(self.config.context_budget_chars)
        latest_result = {}
        try:
            while not state.terminated and state.turn < state.max_turns:
                observation = renderer.render(state, latest_result)
                action = policy.choose_action(observation, state)
                if not isinstance(action, HarnessAction):
                    action = HarnessAction.from_dict(action)
                latest_result = self._execute_harness1_action(action, state)
                state.record_action(action.action.value, action.arguments, latest_result)
                state.turn += 1
            if not state.terminated:
                state.terminated = True
                state.termination_reason = "max_turns"
            trajectory.all_retrieved_chunk_ids = list(state.seen_chunk_ids)
            trajectory.curated_document_ids = state.ordered_curated_ids()
            trajectory.termination_reason = state.termination_reason
            trajectory.action_history = [
                {"turn": event.turn, "action": event.action, "arguments": event.arguments, "result": event.result_summary}
                for event in state.action_history
            ]
            trajectory.sufficiency_events = (
                [{"turn": e.turn, "decision": e.decision, "is_sufficient": e.is_sufficient,
                  "reason": e.reason, "confidence": e.confidence} for e in state.sufficiency.history]
                if state.sufficiency is not None else []
            )
            trajectory.add_stage_log(StageLog(
                stage=StageName.SEARCH_READ,
                input_data={"mode": "harness1", "query": query, "budget": trajectory.budget,
                            "component_fingerprint": self.config.component_fingerprint()},
                output_data={
                    "actions": trajectory.action_history,
                    "episode_state": state.snapshot(),
                    "search_steps_used": len(state.search_history),
                    "max_search_steps": self.config.max_search_steps,
                    "sufficiency_events": trajectory.sufficiency_events,
                },
            ))
            self._apply_curated_context(state)
            self._run_working_memory_stage()
            self._synthesize_from_working_memory(trajectory)
            self._run_verifier_stage()
            self._compute_final_metrics()
        except Exception as error:
            self.console.error(f"Harness-1 episode {query_id} failed: {error}")
            trajectory.final_answer = f"ERROR: {error}"
        trajectory.completed_at = datetime.now()
        self._finalize_trajectory(trajectory)
        self.logger.save_trajectory_summary(trajectory)
        return trajectory

    def _synthesize_from_working_memory(self, trajectory: Trajectory) -> None:
        """H1's final-answer step: the SAME AnswerGenerator/prompt as H0
        (Part F/G), fed from working memory (already restricted to the
        curated set by _apply_curated_context when C2 is enabled) instead
        of the flat transcript. The harness supplies this evidence context
        but never rewrites the model's answer text itself.
        """
        evidence = [
            EvidenceItem(chunk_id=ec.chunk.chunk_id, doc_id=ec.chunk.doc_id, content=ec.chunk.content, score=ec.relevance_score)
            for ec in self.working_memory.get_active_chunks()
        ]
        context = AnswerContext(query=trajectory.query, evidence=evidence)
        start_time = time.time()
        result = self.answer_generator.generate(context)
        duration = (time.time() - start_time) * 1000

        trajectory.final_answer = result.answer
        trajectory.final_cited_chunk_ids = result.cited_chunk_ids
        trajectory.claims = result.claims

        trajectory.add_stage_log(StageLog(
            stage=StageName.SYNTHESIS,
            input_data={"mode": "harness1", "generator": type(self.answer_generator).__name__, "evidence_count": len(evidence)},
            output_data={"answer": result.answer, "cited_chunk_ids": result.cited_chunk_ids, "claims": result.claims, "reasoning": result.reasoning},
            duration_ms=duration,
        ))

    @staticmethod
    def _finalize_trajectory(trajectory: Trajectory) -> None:
        trajectory.total_duration_ms = (trajectory.completed_at - trajectory.started_at).total_seconds() * 1000
        MetricsCalculator.enrich_trajectory(trajectory)

    def _search_budget_remaining(self, state: EpisodeState) -> int:
        return max(0, self.config.max_search_steps - len(state.search_history))

    def _execute_harness1_action(self, action: HarnessAction, state: EpisodeState) -> Dict[str, Any]:
        args = action.arguments
        if action.action == ActionType.FAN_OUT_SEARCH:
            queries = args.get("queries", [])
            if not queries:
                queries = [state.query]
            queries = queries[:5]
            remaining = self._search_budget_remaining(state)
            if remaining <= 0:
                return {"tool": action.action.value, "queries": [], "results": [], "budget_exhausted": True}
            allowed = queries[:remaining]
            results = [self._execute_harness1_search(query, state, "fan_out_search") for query in allowed]
            return {"tool": action.action.value, "queries": allowed, "results": results, "budget_exhausted": len(allowed) < len(queries)}
        if action.action == ActionType.SEARCH_CORPUS:
            if self._search_budget_remaining(state) <= 0:
                return {"tool": action.action.value, "query": args.get("query", state.query), "new_documents": [], "returned": 0, "budget_exhausted": True}
            return self._execute_harness1_search(str(args.get("query", state.query)), state, "search_corpus")
        if action.action == ActionType.GREP_CORPUS:
            if self._search_budget_remaining(state) <= 0:
                return {"tool": action.action.value, "new_documents": [], "budget_exhausted": True}
            pattern = str(args.get("pattern", ""))
            result = self.search_tools.grep_corpus(pattern, int(args.get("max_results", 5)))
            self._ingest_search_result(result, state, "grep_corpus")
            state.search_history.append({"turn": state.turn, "tool": "grep_corpus", "pattern": pattern, "new_documents": self._new_document_ids(result)})
            return {"tool": action.action.value, "new_documents": self._new_document_ids(result)}
        if action.action == ActionType.READ_DOCUMENT:
            doc_id = str(args.get("doc_id", ""))
            result = self.search_tools.read_document(doc_id)
            document = self.corpus_index.get_document(doc_id)
            if document:
                state.add_document(document)
                for chunk in document.chunks:
                    self.working_memory.add_chunks([chunk], "read_document", doc_id, [chunk.score])
            return {"tool": action.action.value, "doc_id": doc_id, "found": document is not None, "chunks": len(result.get("results", []))}
        if action.action == ActionType.REVIEW_DOCS:
            ids = args.get("doc_ids", args.get("ids", []))
            return {"tool": action.action.value, "documents": [state.document_store[item_id].doc_id for item_id in ids if item_id in state.document_store]}
        if action.action == ActionType.CURATE:
            add_ids = args.get("add_ids", [])
            remove_ids = args.get("remove_ids", [])
            importance = args.get("importance", {})
            try:
                return {"tool": action.action.value, **state.curate(add_ids, remove_ids, importance, args.get("rationale", ""))}
            except ComponentUnavailableError as error:
                return {"tool": action.action.value, "error": str(error), "unavailable": True}
        if action.action == ActionType.VERIFY:
            return self._execute_harness1_verify(str(args.get("claim", "")), args.get("doc_ids", []), state)
        if action.action == ActionType.GRAPH_QUERY:
            if state.evidence_graph is None:
                return {"tool": action.action.value, "error": "C3 Evidence Graph is disabled for this episode.", "unavailable": True}
            entity = str(args.get("entity", ""))
            return {"tool": action.action.value, "entity": entity, "documents": state.evidence_graph.graph_query(entity)}
        if action.action == ActionType.GRAPH_NEIGHBORS:
            if state.evidence_graph is None:
                return {"tool": action.action.value, "error": "C3 Evidence Graph is disabled for this episode.", "unavailable": True}
            doc_id = str(args.get("doc_id", ""))
            return {"tool": action.action.value, "doc_id": doc_id, "neighbors": state.evidence_graph.graph_neighbors(doc_id)}
        if action.action == ActionType.CHECK_SUFFICIENCY:
            if state.sufficiency is None:
                return {"tool": action.action.value, "error": "C5 Explicit Sufficiency Check is disabled for this episode.", "unavailable": True}
            event = state.sufficiency.check_sufficiency(self.working_memory, self.current_trajectory, self.current_trajectory.constraints, state.turn)
            return {"tool": action.action.value, "decision": event.decision, "is_sufficient": event.is_sufficient,
                    "reason": event.reason, "confidence": event.confidence}
        if action.action == ActionType.END_SEARCH:
            state.terminated = True
            state.termination_reason = str(args.get("reason", "policy_end"))
            return {"tool": action.action.value, "reason": state.termination_reason, "curated_ids": state.ordered_curated_ids()}
        raise ValueError(f"Unsupported action: {action.action.value}")

    def _execute_harness1_search(self, query: str, state: EpisodeState, source_tool: str) -> Dict[str, Any]:
        result = self.search_tools.search_corpus(query, self.config.max_chunks_per_search)
        new_ids = self._ingest_search_result(result, state, source_tool)
        state.search_history.append({"turn": state.turn, "tool": source_tool, "query": query, "new_documents": new_ids})
        # Auto-seeding is a C1+C2 orchestration policy: the harness may
        # choose to curate the first search's candidates only when both
        # the candidate pool and curated set actually exist. It must never
        # implicitly create either component -- if C1 or C2 is OFF, no
        # auto-seed happens (there is nothing to seed into).
        if (self.config.auto_seed and state.candidate_pool is not None and state.curated_set is not None
                and not state.auto_seeded and new_ids):
            state.curate(new_ids[:8], [], {item_id: "fair" for item_id in new_ids[:8]}, "auto_seed")
            state.auto_seeded = True
        return {"tool": source_tool, "query": query, "new_documents": new_ids, "returned": len(result.get("results", []))}

    def _ingest_search_result(self, result: Dict[str, Any], state: EpisodeState, source_tool: str) -> List[str]:
        """Route a raw search/grep result into whichever of C1/C3 are enabled.

        This never implicitly activates a component: `state.ingest_chunk`
        only touches the candidate pool / evidence graph if they already
        exist (see EpisodeState.ingest_chunk). WorkingMemory is baseline
        H1 evidence bookkeeping, not one of C1-C5, so it is always updated
        here regardless of the C1-C5 configuration.
        """
        new_documents = []
        for chunk_id in result.get("new_chunk_ids", []):
            chunk = self.corpus_index.get_chunk(chunk_id)
            if chunk is not None:
                is_new = state.ingest_chunk(chunk, source_tool)
                self.working_memory.add_chunks([chunk], source_tool, result.get("query", result.get("pattern", "")), [chunk.score])
                if is_new:
                    new_documents.append(chunk.doc_id)
        return new_documents

    @staticmethod
    def _new_document_ids(result: Dict[str, Any]) -> List[str]:
        return list(dict.fromkeys(item.get("doc_id") for item in result.get("results", []) if item.get("doc_id")))

    def _execute_harness1_verify(self, claim: str, doc_ids: List[str], state: EpisodeState) -> Dict[str, Any]:
        """Verify a claim against remembered documents/candidates.

        Verification itself (token-overlap support check) runs
        unconditionally -- it does not depend on C4. The verification
        cache (C4) is consulted and written to only when
        state.verification_cache is not None; with C4 OFF, every call is
        recomputed from scratch and nothing is cached or reused, exactly
        as required.
        """
        records = []
        for item_id in doc_ids:
            cache = state.verification_cache
            if cache is not None:
                cached_record = cache.get(claim, item_id)
                if cached_record is not None:
                    records.append({"doc_id": item_id, "supported": cached_record.supported, "cached": True})
                    continue

            candidate = state.candidate_pool.candidates.get(item_id) if state.candidate_pool is not None else None
            text = ""
            if item_id in state.document_store:
                text = state.document_store[item_id].content
            elif candidate:
                text = candidate.snippet
            claim_terms = set(claim.lower().split())
            text_terms = set(text.lower().split())
            overlap = len(claim_terms & text_terms) / len(claim_terms) if claim_terms else 0.0
            supported = overlap >= self.config.verifier_config.min_claim_overlap

            if cache is not None:
                cache.store(VerificationRecord(claim, item_id, supported, f"token_overlap={overlap:.3f}", state.turn))
            records.append({"doc_id": item_id, "supported": supported, "cached": False, "overlap": overlap})
        return {"tool": ActionType.VERIFY.value, "claim": claim, "records": records}

    def _apply_curated_context(self, state: EpisodeState) -> None:
        """Restrict synthesis context to the curated set, if C2 is enabled.

        With C2 OFF there is no curated set to restrict to, so working
        memory is left untouched -- curation must not silently affect
        rendering/synthesis when the component that would produce it does
        not exist.
        """
        if state.curated_set is None:
            return
        curated_ids = set(state.curated_set.curated)
        uncurated_chunk_ids = [
            evidence.chunk.chunk_id
            for evidence in self.working_memory.get_active_chunks()
            if evidence.chunk.doc_id not in curated_ids
        ]
        self.working_memory.prune_chunks(uncurated_chunk_ids, "not in final Harness-1 curated set")


    def _run_planner_stage(self):
        self.console.stage_start(StageName.PLANNER, self.current_trajectory.query_id)
        start_time = time.time()

        input_data = {"query": self.current_trajectory.query}
        plan = self.planner.plan(self.current_trajectory.query)

        self.current_trajectory.constraints = [
            Constraint(type=c.type, description=c.description, raw_text=c.raw_text, parsed_value=c.parsed_value)
            for c in plan.constraints
        ]
        self.plan = plan

        output_data = {
            "constraints": [
                {"type": c.type, "description": c.description, "raw_text": c.raw_text, "parsed_value": c.parsed_value}
                for c in self.current_trajectory.constraints
            ],
            "sub_queries": [
                {"id": sq.id, "text": sq.text, "dependencies": sq.dependencies, "type": sq.expected_answer_type}
                for sq in plan.sub_queries
            ],
            "reasoning": plan.reasoning
        }

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.PLANNER, input_data, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.PLANNER, input_data=input_data, output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.PLANNER, duration)
        self.console.stage_output(StageName.PLANNER, f"Extracted {len(self.current_trajectory.constraints)} constraints, {len(plan.sub_queries)} sub-queries")

    def _run_search_read_loop(self):
        self.console.stage_start(StageName.SEARCH_READ, self.current_trajectory.query_id)
        start_time = time.time()

        all_search_results = []
        search_steps_used = 0
        budget_exhausted = False
        max_search_steps = self.config.max_search_steps

        for sub_query in self.plan.sub_queries:
            if search_steps_used >= max_search_steps:
                budget_exhausted = True
                break

            self.console.info(f"Processing sub-query: {sub_query.text}")

            search_result = self.search_tools.search_corpus(
                sub_query.text,
                top_k=self.config.max_chunks_per_search
            )
            all_search_results.append(search_result)
            search_steps_used += 1

            chunks = [self.corpus_index.get_chunk(cid) for cid in search_result["new_chunk_ids"]]
            chunks = [c for c in chunks if c is not None]
            if chunks:
                self.working_memory.add_chunks(
                    chunks,
                    source_tool="search_corpus",
                    query_or_pattern=sub_query.text,
                    relevance_scores=[c.score for c in chunks]
                )

            if search_steps_used >= max_search_steps:
                budget_exhausted = True
                break

            if self._should_grep(sub_query.text):
                grep_pattern = self._extract_grep_pattern(sub_query.text)
                if grep_pattern:
                    grep_result = self.search_tools.grep_corpus(grep_pattern, max_results=5)
                    all_search_results.append(grep_result)
                    search_steps_used += 1

                    grep_chunks = [self.corpus_index.get_chunk(cid) for cid in grep_result["new_chunk_ids"]]
                    grep_chunks = [c for c in grep_chunks if c is not None]
                    if grep_chunks:
                        self.working_memory.add_chunks(
                            grep_chunks,
                            source_tool="grep_corpus",
                            query_or_pattern=grep_pattern
                        )

        self.current_trajectory.all_retrieved_chunk_ids = self.search_tools.get_seen_chunk_ids()
        self.current_trajectory.search_calls = search_steps_used
        if budget_exhausted:
            self.current_trajectory.termination_reason = "max_search_steps"

        output_data = {
            "search_results": all_search_results,
            "total_chunks_retrieved": len(self.current_trajectory.all_retrieved_chunk_ids),
            "working_memory_stats": self.working_memory.get_stats(),
            "search_steps_used": search_steps_used,
            "max_search_steps": max_search_steps,
            "budget_exhausted": budget_exhausted,
        }

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.SEARCH_READ, {"sub_queries": [sq.text for sq in self.plan.sub_queries]}, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.SEARCH_READ, input_data={"sub_queries": [sq.text for sq in self.plan.sub_queries]},
            output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.SEARCH_READ, duration)
        self.console.stage_output(StageName.SEARCH_READ, f"Retrieved {len(self.current_trajectory.all_retrieved_chunk_ids)} unique chunks")

    def _should_grep(self, query: str) -> bool:
        import re
        return bool(re.search(r'\b[A-Z][a-z]+\b|\b\d{4}\b|\b\d+\b', query))

    def _extract_grep_pattern(self, query: str) -> Optional[str]:
        import re
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        years = re.findall(r'\b\d{4}\b', query)

        if proper_nouns:
            return proper_nouns[0]
        if years:
            return years[0]
        return None

    def _run_working_memory_stage(self):
        self.console.stage_start(StageName.WORKING_MEMORY, self.current_trajectory.query_id)
        start_time = time.time()

        stats = self.working_memory.get_stats()
        active_chunks = self.working_memory.get_active_chunks()

        input_data = {"action": "log_state"}
        output_data = {
            "stats": stats,
            "active_chunk_ids": [ec.chunk.chunk_id for ec in active_chunks],
            "pruned_chunk_ids": [ec.chunk.chunk_id for ec in self.working_memory.get_all_chunks() if ec.is_pruned]
        }

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.WORKING_MEMORY, input_data, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.WORKING_MEMORY, input_data=input_data, output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.WORKING_MEMORY, duration)
        self.console.stage_output(StageName.WORKING_MEMORY, f"Active chunks: {stats['active_chunks']}, Pruned: {stats['pruned_chunks']}")

    def _run_sufficiency_check_stage(self):
        self.console.stage_start(StageName.SUFFICIENCY_CHECK, self.current_trajectory.query_id)
        start_time = time.time()

        input_data = {
            "working_memory_stats": self.working_memory.get_stats(),
            "constraints": [{"type": c.type, "description": c.description} for c in self.current_trajectory.constraints]
        }

        sufficiency_result = self.sufficiency_checker.check(
            self.working_memory,
            self.current_trajectory,
            self.current_trajectory.constraints
        )

        output_data = {
            "is_sufficient": sufficiency_result.is_sufficient,
            "reason": sufficiency_result.reason,
            "missing_info": sufficiency_result.missing_info,
            "confidence": sufficiency_result.confidence
        }

        self.current_trajectory.sufficiency_decision = sufficiency_result.is_sufficient
        self.current_trajectory.sufficiency_reason = sufficiency_result.reason

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.SUFFICIENCY_CHECK, input_data, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.SUFFICIENCY_CHECK, input_data=input_data, output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.SUFFICIENCY_CHECK, duration)
        self.console.stage_output(StageName.SUFFICIENCY_CHECK, f"Sufficient: {sufficiency_result.is_sufficient} (confidence: {sufficiency_result.confidence:.2f})")

    def _run_synthesis_stage(self):
        self.console.stage_start(StageName.SYNTHESIS, self.current_trajectory.query_id)
        start_time = time.time()

        input_data = {
            "query": self.current_trajectory.query,
            "working_memory_context": self.working_memory.get_context_for_synthesis(),
            "sufficiency_decision": self.current_trajectory.sufficiency_decision
        }

        synthesis_result = self.synthesis_engine.synthesize(
            self.current_trajectory.query,
            self.working_memory,
            self.current_trajectory,
            self.current_trajectory.constraints
        )

        output_data = {
            "answer": synthesis_result.answer,
            "claims": [
                {
                    "text": c.text,
                    "citations": c.citation_chunk_ids,
                    "confidence": c.confidence,
                    "type": c.claim_type
                }
                for c in synthesis_result.claims
            ],
            "cited_chunk_ids": synthesis_result.cited_chunk_ids,
            "reasoning": synthesis_result.reasoning
        }

        self.current_trajectory.final_answer = synthesis_result.answer
        self.current_trajectory.final_cited_chunk_ids = synthesis_result.cited_chunk_ids
        self.current_trajectory.claims = output_data["claims"]

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.SYNTHESIS, input_data, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.SYNTHESIS, input_data=input_data, output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.SYNTHESIS, duration)
        self.console.stage_output(StageName.SYNTHESIS, f"Answer length: {len(synthesis_result.answer)}, Citations: {len(synthesis_result.cited_chunk_ids)}")

    def _run_verifier_stage(self):
        self.console.stage_start(StageName.VERIFIER, self.current_trajectory.query_id)
        start_time = time.time()

        from .synthesis import Claim
        synthesis_log = next(
            (log for log in reversed(self.current_trajectory.stage_logs) if log.stage == StageName.SYNTHESIS),
            None,
        )
        claims = [
            Claim(text=c["text"], citation_chunk_ids=c["citations"], confidence=c["confidence"], claim_type=c["type"])
            for c in (synthesis_log.output_data.get("claims", []) if synthesis_log else [])
        ]

        from .synthesis import SynthesisResult
        synthesis_result = SynthesisResult(
            answer=self.current_trajectory.final_answer,
            claims=claims,
            cited_chunk_ids=self.current_trajectory.final_cited_chunk_ids
        )

        input_data = {
            "final_answer": self.current_trajectory.final_answer,
            "cited_chunk_ids": self.current_trajectory.final_cited_chunk_ids
        }

        verification_result = self.verifier.verify(synthesis_result, self.working_memory, self.current_trajectory)

        output_data = {
            "all_citations_valid": verification_result.all_citations_valid,
            "invalid_citations": verification_result.invalid_citations,
            "claims_verified": verification_result.claims_verified
        }
        self.current_trajectory.verification = output_data

        duration = (time.time() - start_time) * 1000
        self.logger.log_stage(StageName.VERIFIER, input_data, output_data, duration)
        self.current_trajectory.add_stage_log(StageLog(
            stage=StageName.VERIFIER, input_data=input_data, output_data=output_data, duration_ms=duration
        ))

        self.console.stage_complete(StageName.VERIFIER, duration)
        self.console.stage_output(StageName.VERIFIER, f"All valid: {verification_result.all_citations_valid}, Invalid: {len(verification_result.invalid_citations)}")

    def _compute_final_metrics(self):
        metrics = calculate_recall_metrics(self.current_trajectory)
        self.current_trajectory.trajectory_recall = metrics.trajectory_recall
        self.current_trajectory.output_recall = metrics.output_recall

        self.console.info(f"Trajectory Recall: {metrics.trajectory_recall:.3f}")
        self.console.info(f"Output Recall: {metrics.output_recall:.3f}")
        self.console.info(f"Failure Mode: {metrics.failure_mode}")

    def run_benchmark(self, queries: List[Dict[str, Any]], benchmark_name: str) -> BenchmarkResult:
        result = BenchmarkResult(benchmark=benchmark_name, model_name=self.config.model_name)

        for q in queries:
            query_id = q.get("query_id", str(uuid.uuid4())[:8])
            query = q["query"]
            gold_chunks = q.get("gold_chunk_ids", [])
            task_id = q.get("task_id", query_id)
            trial_id = str(q.get("trial_id", "0"))
            condition = q.get("condition")

            trajectory = self.run_trajectory(query, query_id, gold_chunks, benchmark_name,
                                              task_id=task_id, trial_id=trial_id, condition=condition)
            result.trajectories.append(trajectory)

        result.compute_aggregates()
        self.logger.save_benchmark_result(result)

        return result


def create_harness(corpus_index: CorpusIndex, config: HarnessConfig = None,
                   log_dir: str = "harness/logs") -> DeepResearchHarness:
    return DeepResearchHarness(corpus_index, config, log_dir)