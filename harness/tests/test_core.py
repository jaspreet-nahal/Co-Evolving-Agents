import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from harness.core import InMemoryCorpusIndex, CorpusDocument, CorpusChunk, SearchReadTools, WorkingMemory, EvidenceChunk, Planner, Constraint, ConstraintType, Plan, SubQuery, SufficiencyChecker, SufficiencyConfig, SufficiencyCriterion, SynthesisEngine, SynthesisResult, Claim, Verifier, VerificationConfig, DeepResearchHarness, HarnessConfig, Trajectory, StageName, StageLog, calculate_recall_metrics, RecallMetrics
from datetime import datetime


class TestCorpusIndex:

    def test_add_and_search(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=20)
        doc = CorpusDocument(
            doc_id="test_1",
            content="This is a test document about machine learning and AI."
        )
        index.add_document(doc)

        assert len(index.documents) == 1
        assert len(index.chunks) > 0
        assert "test_1" in index.documents

    def test_search_returns_chunks(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(
            doc_id="test_1",
            content="Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."
        )
        index.add_document(doc)

        results = index.search("machine learning", top_k=5, exclude_chunk_ids=set())
        assert len(results) > 0
        assert all(isinstance(c, CorpusChunk) for c in results)

    def test_grep_returns_chunks(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(
            doc_id="test_1",
            content="Machine learning is a subset of artificial intelligence."
        )
        index.add_document(doc)

        results = index.grep(r"machine", max_results=5, exclude_chunk_ids=set())
        assert len(results) > 0


class TestSearchReadTools:

    def test_search_corpus_tracks_seen(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Machine learning and AI are related fields.")
        index.add_document(doc)

        tools = SearchReadTools(index)
        result1 = tools.search_corpus("machine learning", top_k=5)
        result2 = tools.search_corpus("machine learning", top_k=5)

        assert len(result1["results"]) > 0
        assert len(result2["results"]) == 0
        assert result2["new_chunk_ids"] == []

    def test_grep_corpus(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Python is a programming language.")
        index.add_document(doc)

        tools = SearchReadTools(index)
        result = tools.grep_corpus(r"Python", max_results=5)

        assert len(result["results"]) > 0

    def test_read_document(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Test document content.")
        index.add_document(doc)

        tools = SearchReadTools(index)
        result = tools.read_document("d1")

        assert len(result["results"]) > 0
        assert result["results"][0]["doc_id"] == "d1"

    def test_prune_chunks(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Test content for pruning.")
        index.add_document(doc)

        tools = SearchReadTools(index)
        tools.search_corpus("test", top_k=5)
        chunk_ids = tools.get_seen_chunk_ids()
        assert len(chunk_ids) > 0

        result = tools.prune_chunks(chunk_ids)
        assert len(result["pruned_chunk_ids"]) > 0
        assert len(tools.get_seen_chunk_ids()) > 0


class TestWorkingMemory:

    def test_add_chunks(self):
        wm = WorkingMemory()
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Evidence chunk one. Evidence chunk two.")
        index.add_document(doc)

        chunks = list(index.chunks.values())[:2]
        added = wm.add_chunks(chunks, "search_corpus", "test query", [0.9, 0.8])

        assert len(added) == 2
        assert wm.get_stats()["active_chunks"] == 2

    def test_prune_chunks(self):
        wm = WorkingMemory()
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Chunk one. Chunk two. Chunk three.")
        index.add_document(doc)

        chunks = list(index.chunks.values())
        wm.add_chunks(chunks, "search", "query")

        pruned = wm.prune_chunks([chunks[0].chunk_id], "low relevance")
        assert len(pruned) == 1
        assert wm.get_stats()["active_chunks"] == 2
        assert wm.get_stats()["pruned_chunks"] == 1

    def test_context_for_synthesis(self):
        wm = WorkingMemory()
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Important evidence for the answer.")
        index.add_document(doc)

        chunks = list(index.chunks.values())
        wm.add_chunks(chunks, "search", "query", [0.9])

        context = wm.get_context_for_synthesis()
        assert "CHUNK_ID" in context
        assert "Important evidence" in context


class TestPlanner:

    def test_extract_temporal_constraints(self):
        planner = Planner()
        query = "What happened before 2020?"
        plan = planner.plan(query)

        temporal_constraints = [c for c in plan.constraints if c.type == ConstraintType.TEMPORAL.value]
        assert len(temporal_constraints) > 0
        assert "before 2020" in temporal_constraints[0].raw_text.lower()

    def test_extract_exclusion_constraints(self):
        planner = Planner()
        query = "List companies excluding subsidiaries."
        plan = planner.plan(query)

        exclusion_constraints = [c for c in plan.constraints if c.type == ConstraintType.ENTITY_EXCLUSION.value]
        assert len(exclusion_constraints) > 0
        assert "excluding" in exclusion_constraints[0].raw_text.lower()

    def test_decompose_query(self):
        planner = Planner()
        query = "What is AI and how does it work?"
        plan = planner.plan(query)

        assert len(plan.sub_queries) >= 1
        assert plan.sub_queries[0].text == query


class TestSufficiencyChecker:

    def test_basic_sufficiency(self):
        config = SufficiencyConfig(criteria=[SufficiencyCriterion.ANY_ANSWER], min_evidence_chunks=1)
        checker = SufficiencyChecker(config)

        wm = WorkingMemory()
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="Some evidence.")
        index.add_document(doc)
        chunks = list(index.chunks.values())
        wm.add_chunks(chunks, "search", "query")

        trajectory = Trajectory(
            query_id="test",
            query="test query",
            model_name="test",
            benchmark="test"
        )

        result = checker.check(wm, trajectory)
        assert result.is_sufficient == True

    def test_insufficient_evidence(self):
        config = SufficiencyConfig(criteria=[SufficiencyCriterion.EVIDENCE_THRESHOLD], min_evidence_chunks=5)
        checker = SufficiencyChecker(config)

        wm = WorkingMemory()
        trajectory = Trajectory(query_id="test", query="test", model_name="test", benchmark="test")

        result = checker.check(wm, trajectory)
        assert result.is_sufficient == False
        assert "Insufficient evidence" in result.reason


class TestSynthesisEngine:

    def test_synthesize_empty_memory(self):
        engine = SynthesisEngine()
        wm = WorkingMemory()
        trajectory = Trajectory(query_id="test", query="test", model_name="test", benchmark="test")

        result = engine.synthesize("test query", wm, trajectory)
        assert "Insufficient evidence" in result.answer
        assert len(result.cited_chunk_ids) == 0

    def test_synthesize_with_evidence(self):
        engine = SynthesisEngine()
        wm = WorkingMemory()
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="The answer is 42.")
        index.add_document(doc)
        chunks = list(index.chunks.values())
        wm.add_chunks(chunks, "search", "query", [0.9])

        trajectory = Trajectory(query_id="test", query="What is the answer?", model_name="test", benchmark="test")

        result = engine.synthesize("What is the answer?", wm, trajectory)
        assert len(result.cited_chunk_ids) > 0
        assert "42" in result.answer


class TestVerifier:

    def test_verify_valid_citation(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)
        doc = CorpusDocument(doc_id="d1", content="The answer is 42.")
        index.add_document(doc)

        verifier = Verifier(index)
        wm = WorkingMemory()
        chunks = list(index.chunks.values())
        wm.add_chunks(chunks, "search", "query")

        from harness.core import Claim
        claim = Claim(text="The answer is 42", citation_chunk_ids=[chunks[0].chunk_id])
        result = verifier.verify(
            SynthesisResult(answer="The answer is 42", claims=[claim], cited_chunk_ids=[chunks[0].chunk_id]),
            wm,
            Trajectory(query_id="test", query="test", model_name="test", benchmark="test")
        )

        assert result.all_citations_valid == True
        assert len(result.invalid_citations) == 0

    def test_verify_invalid_citation(self):
        index = InMemoryCorpusIndex(chunk_size=50, chunk_overlap=10)

        verifier = Verifier(index)
        wm = WorkingMemory()

        from harness.core import Claim
        claim = Claim(text="Something", citation_chunk_ids=["nonexistent_chunk"])
        result = verifier.verify(
            SynthesisResult(answer="Something", claims=[claim], cited_chunk_ids=["nonexistent_chunk"]),
            wm,
            Trajectory(query_id="test", query="test", model_name="test", benchmark="test")
        )

        assert result.all_citations_valid == False
        assert len(result.invalid_citations) > 0


class TestMetrics:

    def test_perfect_recall(self):
        trajectory = Trajectory(
            query_id="test",
            query="test",
            model_name="test",
            benchmark="test",
            all_retrieved_chunk_ids=["c1", "c2", "c3"],
            final_cited_chunk_ids=["c1", "c2"],
            gold_relevant_chunk_ids=["c1", "c2"]
        )

        metrics = calculate_recall_metrics(trajectory)
        assert metrics.trajectory_recall == 1.0
        assert metrics.output_recall == 1.0
        assert metrics.failure_mode == "success"

    def test_found_not_used(self):
        trajectory = Trajectory(
            query_id="test",
            query="test",
            model_name="test",
            benchmark="test",
            all_retrieved_chunk_ids=["c1", "c2", "c3"],
            final_cited_chunk_ids=["c1"],
            gold_relevant_chunk_ids=["c1", "c2"]
        )

        metrics = calculate_recall_metrics(trajectory)
        assert metrics.trajectory_recall == 1.0
        assert metrics.output_recall == 0.5
        assert metrics.failure_mode == "found_not_used"

    def test_never_found(self):
        trajectory = Trajectory(
            query_id="test",
            query="test",
            model_name="test",
            benchmark="test",
            all_retrieved_chunk_ids=["c3", "c4"],
            final_cited_chunk_ids=["c3"],
            gold_relevant_chunk_ids=["c1", "c2"]
        )

        metrics = calculate_recall_metrics(trajectory)
        assert metrics.trajectory_recall == 0.0
        assert metrics.output_recall == 0.0
        assert metrics.failure_mode == "never_found"


class TestHarnessIntegration:

    def test_full_trajectory(self):
        index = InMemoryCorpusIndex(chunk_size=100, chunk_overlap=20)
        docs = [
            CorpusDocument(doc_id="d1", content="Apple revenue was $394B in 2022."),
            CorpusDocument(doc_id="d2", content="Microsoft revenue was $211B in 2023."),
        ]
        for doc in docs:
            index.add_document(doc)

        config = HarnessConfig(
            max_search_steps=2,
            max_chunks_per_search=5,
            model_name="test_model"
        )
        harness = DeepResearchHarness(index, config)

        trajectory = harness.run_trajectory(
            query="What was Apple's revenue in 2022?",
            query_id="test_1",
            gold_chunk_ids=["d1_chunk_0"],
            benchmark="test"
        )

        assert trajectory.completed_at is not None
        assert len(trajectory.stage_logs) == 6
        stages = [log.stage for log in trajectory.stage_logs]
        assert StageName.PLANNER in stages
        assert StageName.SEARCH_READ in stages
        assert StageName.WORKING_MEMORY in stages
        assert StageName.SUFFICIENCY_CHECK in stages
        assert StageName.SYNTHESIS in stages
        assert StageName.VERIFIER in stages

        assert trajectory.trajectory_recall >= 0
        assert trajectory.output_recall >= 0

        assert trajectory.sufficiency_decision is not None
        assert trajectory.sufficiency_reason != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])