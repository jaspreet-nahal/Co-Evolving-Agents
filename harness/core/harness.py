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
from .episode import EpisodeState, VerificationRecord
from .model_adapter import RuleBasedPolicy
from .observations import ObservationRenderer


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

        self.current_trajectory: Optional[Trajectory] = None
        self.plan: Optional[Plan] = None

    def run_trajectory(self, query: str, query_id: str = None,
                       gold_chunk_ids: List[str] = None,
                       benchmark: str = "unknown") -> Trajectory:
        if self.config.execution_mode == "harness1":
            return self.run_harness1_episode(query, query_id, gold_chunk_ids, benchmark)
        query_id = query_id or str(uuid.uuid4())[:8]
        gold_chunk_ids = gold_chunk_ids or []

        self.current_trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
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

    def run_harness1_episode(self, query: str, query_id: str = None,
                             gold_chunk_ids: List[str] = None,
                             benchmark: str = "unknown") -> Trajectory:
        query_id = query_id or str(uuid.uuid4())[:8]
        trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
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
            trajectory.add_stage_log(StageLog(
                stage=StageName.SEARCH_READ,
                input_data={"mode": "harness1", "query": query},
                output_data={"actions": trajectory.action_history, "episode_state": state.snapshot()},
            ))
            self._apply_curated_context(state)
            self._run_working_memory_stage()
            self._run_sufficiency_check_stage()
            self._run_synthesis_stage()
            self._run_verifier_stage()
            self._compute_final_metrics()
        except Exception as error:
            self.console.error(f"Harness-1 episode {query_id} failed: {error}")
            trajectory.final_answer = f"ERROR: {error}"
        trajectory.completed_at = datetime.now()
        self._finalize_trajectory(trajectory)
        self.logger.save_trajectory_summary(trajectory)
        return trajectory

    @staticmethod
    def _finalize_trajectory(trajectory: Trajectory) -> None:
        trajectory.total_duration_ms = (trajectory.completed_at - trajectory.started_at).total_seconds() * 1000
        MetricsCalculator.enrich_trajectory(trajectory)

    def _execute_harness1_action(self, action: HarnessAction, state: EpisodeState) -> Dict[str, Any]:
        args = action.arguments
        if action.action == ActionType.FAN_OUT_SEARCH:
            queries = args.get("queries", [])
            if not queries:
                queries = [state.query]
            results = [self._execute_harness1_search(query, state, "fan_out_search") for query in queries[:5]]
            return {"tool": action.action.value, "queries": queries[:5], "results": results}
        if action.action == ActionType.SEARCH_CORPUS:
            return self._execute_harness1_search(str(args.get("query", state.query)), state, "search_corpus")
        if action.action == ActionType.GREP_CORPUS:
            pattern = str(args.get("pattern", ""))
            result = self.search_tools.grep_corpus(pattern, int(args.get("max_results", 5)))
            self._ingest_search_result(result, state, "grep_corpus")
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
            return {"tool": action.action.value, **state.curate(add_ids, remove_ids, importance, args.get("rationale", ""))}
        if action.action == ActionType.VERIFY:
            return self._execute_harness1_verify(str(args.get("claim", "")), args.get("doc_ids", []), state)
        if action.action == ActionType.END_SEARCH:
            state.terminated = True
            state.termination_reason = str(args.get("reason", "policy_end"))
            return {"tool": action.action.value, "reason": state.termination_reason, "curated_ids": state.ordered_curated_ids()}
        raise ValueError(f"Unsupported action: {action.action.value}")

    def _execute_harness1_search(self, query: str, state: EpisodeState, source_tool: str) -> Dict[str, Any]:
        result = self.search_tools.search_corpus(query, self.config.max_chunks_per_search)
        new_ids = self._ingest_search_result(result, state, source_tool)
        state.search_history.append({"turn": state.turn, "tool": source_tool, "query": query, "new_documents": new_ids})
        if self.config.auto_seed and not state.auto_seeded and new_ids:
            state.curate(new_ids[:8], [], {item_id: "fair" for item_id in new_ids[:8]}, "auto_seed")
            state.auto_seeded = True
        return {"tool": source_tool, "query": query, "new_documents": new_ids, "returned": len(result.get("results", []))}

    def _ingest_search_result(self, result: Dict[str, Any], state: EpisodeState, source_tool: str) -> List[str]:
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
        records = []
        for item_id in doc_ids:
            key = state.cache_key(claim, item_id)
            if key in state.verification_cache:
                record = state.verification_cache[key]
                records.append({"doc_id": item_id, "supported": record.supported, "cached": True})
                continue
            candidate = state.candidates.get(item_id)
            text = ""
            if item_id in state.document_store:
                text = state.document_store[item_id].content
            elif candidate:
                text = candidate.snippet
            claim_terms = set(claim.lower().split())
            text_terms = set(text.lower().split())
            overlap = len(claim_terms & text_terms) / len(claim_terms) if claim_terms else 0.0
            supported = overlap >= self.config.verifier_config.min_claim_overlap
            state.verification_cache[key] = VerificationRecord(claim, item_id, supported, f"token_overlap={overlap:.3f}", state.turn)
            records.append({"doc_id": item_id, "supported": supported, "cached": False, "overlap": overlap})
        return {"tool": ActionType.VERIFY.value, "claim": claim, "records": records}

    def _apply_curated_context(self, state: EpisodeState) -> None:
        curated_ids = set(state.curated)
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

        for sub_query in self.plan.sub_queries:
            self.console.info(f"Processing sub-query: {sub_query.text}")

            search_result = self.search_tools.search_corpus(
                sub_query.text,
                top_k=self.config.max_chunks_per_search
            )
            all_search_results.append(search_result)

            chunks = [self.corpus_index.get_chunk(cid) for cid in search_result["new_chunk_ids"]]
            chunks = [c for c in chunks if c is not None]
            if chunks:
                self.working_memory.add_chunks(
                    chunks,
                    source_tool="search_corpus",
                    query_or_pattern=sub_query.text,
                    relevance_scores=[c.score for c in chunks]
                )

            if self._should_grep(sub_query.text):
                grep_pattern = self._extract_grep_pattern(sub_query.text)
                if grep_pattern:
                    grep_result = self.search_tools.grep_corpus(grep_pattern, max_results=5)
                    all_search_results.append(grep_result)

                    grep_chunks = [self.corpus_index.get_chunk(cid) for cid in grep_result["new_chunk_ids"]]
                    grep_chunks = [c for c in grep_chunks if c is not None]
                    if grep_chunks:
                        self.working_memory.add_chunks(
                            grep_chunks,
                            source_tool="grep_corpus",
                            query_or_pattern=grep_pattern
                        )

        self.current_trajectory.all_retrieved_chunk_ids = self.search_tools.get_seen_chunk_ids()

        output_data = {
            "search_results": all_search_results,
            "total_chunks_retrieved": len(self.current_trajectory.all_retrieved_chunk_ids),
            "working_memory_stats": self.working_memory.get_stats()
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

            trajectory = self.run_trajectory(query, query_id, gold_chunks, benchmark_name)
            result.trajectories.append(trajectory)

        result.compute_aggregates()
        self.logger.save_benchmark_result(result)

        return result


def create_harness(corpus_index: CorpusIndex, config: HarnessConfig = None,
                   log_dir: str = "harness/logs") -> DeepResearchHarness:
    return DeepResearchHarness(corpus_index, config, log_dir)