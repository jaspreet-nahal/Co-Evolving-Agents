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
from .metrics import calculate_recall_metrics, RecallMetrics


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
        query_id = query_id or str(uuid.uuid4())[:8]
        gold_chunk_ids = gold_chunk_ids or []

        self.current_trajectory = Trajectory(
            query_id=query_id,
            query=query,
            model_name=self.config.model_name,
            benchmark=benchmark,
            gold_relevant_chunk_ids=gold_chunk_ids
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
            self.logger.save_trajectory_summary(self.current_trajectory)

        except Exception as e:
            self.console.error(f"Trajectory {query_id} failed: {e}")
            self.current_trajectory.final_answer = f"ERROR: {str(e)}"
            self.logger.save_trajectory_summary(self.current_trajectory)

        self.current_trajectory.completed_at = datetime.now()
        return self.current_trajectory

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
        claims = [
            Claim(text=c["text"], citation_chunk_ids=c["citations"], confidence=c["confidence"], claim_type=c["type"])
            for c in self.current_trajectory.stage_logs[-2].output_data.get("claims", [])
        ] if len(self.current_trajectory.stage_logs) >= 2 else []

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