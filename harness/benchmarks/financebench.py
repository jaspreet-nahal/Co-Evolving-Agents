import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from ..core.models import Trajectory
from ..core.harness import DeepResearchHarness, HarnessConfig
from ..core.metrics import RecallMetrics
from ..core.sufficiency_check import SufficiencyConfig, SufficiencyCriterion
from ..core.search_read import to_corpus_document

@dataclass
class FinanceBenchExample:
    question_id: str
    question: str
    answer: str
    gold_chunk_ids: List[str] = field(default_factory=list)
    doc_id: str = ""
    question_type: str = ""


class RealDataUnavailableError(RuntimeError):
    """Raised in mode='real' when a verified real corpus/dataset cannot be
    loaded. Real benchmark execution must never silently substitute sample
    data; callers must fix the data path/HF access or fall back to
    mode='test' explicitly, never implicitly."""


class FinanceBenchBenchmark:
    def __init__(self, data_dir: str = None, corpus_dir: str = None, mode: str = "test"):
        if mode not in ("test", "real"):
            raise ValueError(f"mode must be 'test' or 'real', got {mode!r}")
        self.data_dir = Path(data_dir) if data_dir else Path("data/financebench")
        self.corpus_dir = Path(corpus_dir) if corpus_dir else Path("data/financebench_corpus")
        self.mode = mode
        self.examples: List[FinanceBenchExample] = []
        self.corpus_index = None

    def load_data(self, max_examples: int = None, split: str = "train") -> List[FinanceBenchExample]:
        local_files = list(self.data_dir.glob("*.jsonl")) + list(self.data_dir.glob("*.json"))
        if local_files:
            return self._load_from_file(local_files[0], max_examples)

        try:
            return self._load_from_hf(max_examples, split)
        except Exception as e:
            if self.mode == "real":
                raise RealDataUnavailableError(
                    f"FinanceBench real-mode data load failed: no local files under {self.data_dir} "
                    f"and the Hugging Face dataset could not be loaded ({e}). Real benchmark "
                    f"execution must never silently use sample data; fix the data source or use mode='test'."
                ) from e
            print(f"Could not load from HF: {e}")

        print("Using sample FinanceBench data for testing (mode='test'; only 150 public questions available in real mode)")
        return self._create_sample_data(max_examples or 20)

    def _load_from_file(self, path: Path, max_examples: int = None) -> List[FinanceBenchExample]:
        examples = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                data = json.loads(line)
                ex = FinanceBenchExample(
                    question_id=data.get("id", data.get("question_id", f"fb_{i}")),
                    question=data.get("question", ""),
                    answer=data.get("answer", data.get("gold_answer", "")),
                    gold_chunk_ids=data.get("evidence", data.get("gold_chunk_ids", [])),
                    doc_id=data.get("doc_id", data.get("filename", "")),
                    question_type=data.get("question_type", data.get("type", ""))
                )
                examples.append(ex)
        self.examples = examples
        return examples

    def _load_from_hf(self, max_examples: int = None, split: str = "train") -> List[FinanceBenchExample]:
        from datasets import load_dataset

        ds = load_dataset("PatronusAI/financebench", split=split)

        examples = []
        for i, item in enumerate(ds):
            if max_examples and i >= max_examples:
                break

            ex = FinanceBenchExample(
                question_id=item.get("id", item.get("question_id", f"fb_{i}")),
                question=item.get("question", ""),
                answer=item.get("answer", item.get("gold_answer", "")),
                gold_chunk_ids=item.get("evidence", item.get("gold_chunk_ids", [])),
                doc_id=item.get("doc_id", item.get("filename", "")),
                question_type=item.get("question_type", item.get("type", ""))
            )
            examples.append(ex)

        self.examples = examples
        return examples

    def _create_sample_data(self, num_examples: int) -> List[FinanceBenchExample]:
        sample_questions = [
            {
                "question_id": "fb_1",
                "question": "What was Apple's total revenue in FY2022?",
                "answer": "$394.33 billion",
                "gold_chunk_ids": ["apple_10k_2022_chunk_0"],
                "doc_id": "apple_10k_2022.pdf",
                "question_type": "extraction"
            },
            {
                "question_id": "fb_2",
                "question": "What is Microsoft's gross margin for FY2023?",
                "answer": "68.9%",
                "gold_chunk_ids": ["msft_10k_2023_chunk_1"],
                "doc_id": "msft_10k_2023.pdf",
                "question_type": "extraction"
            },
            {
                "question_id": "fb_3",
                "question": "How much did Amazon spend on R&D in 2022?",
                "answer": "$73.2 billion",
                "gold_chunk_ids": ["amzn_10k_2022_chunk_2"],
                "doc_id": "amzn_10k_2022.pdf",
                "question_type": "extraction"
            },
            {
                "question_id": "fb_4",
                "question": "Compare the debt-to-equity ratios of Apple and Microsoft for 2023.",
                "answer": "Apple: 1.73, Microsoft: 0.45. Microsoft has a lower debt-to-equity ratio.",
                "gold_chunk_ids": ["apple_10k_2023_chunk_0", "msft_10k_2023_chunk_0"],
                "doc_id": "comparison",
                "question_type": "reasoning"
            },
            {
                "question_id": "fb_5",
                "question": "What was Tesla's free cash flow in Q4 2023?",
                "answer": "$2.06 billion",
                "gold_chunk_ids": ["tsla_10q_q4_2023_chunk_1"],
                "doc_id": "tsla_10q_q4_2023.pdf",
                "question_type": "extraction"
            }
        ]

        examples = []
        for i in range(min(num_examples, len(sample_questions))):
            q = sample_questions[i]
            ex = FinanceBenchExample(
                question_id=q["question_id"],
                question=q["question"],
                answer=q["answer"],
                gold_chunk_ids=q["gold_chunk_ids"],
                doc_id=q["doc_id"],
                question_type=q["question_type"]
            )
            examples.append(ex)

        while len(examples) < num_examples:
            base = sample_questions[len(examples) % len(sample_questions)]
            ex = FinanceBenchExample(
                question_id=f"{base['question_id']}_v{len(examples)}",
                question=base["question"],
                answer=base["answer"],
                gold_chunk_ids=base["gold_chunk_ids"],
                doc_id=base["doc_id"],
                question_type=base["question_type"]
            )
            examples.append(ex)

        self.examples = examples
        return examples

    def load_corpus(self, corpus_path: str = None):
        from ..core.search_read import InMemoryCorpusIndex

        self.corpus_index = InMemoryCorpusIndex(chunk_size=1024, chunk_overlap=100)

        if corpus_path:
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(
                    f"FinanceBench corpus_path '{corpus_path}' was requested but does not exist. "
                    f"Refusing to silently substitute sample data for an explicitly requested corpus."
                )
            self.corpus_index.load_from_jsonl(corpus_path)
        elif self.corpus_dir.exists():
            for file_path in self.corpus_dir.glob("*.jsonl"):
                self.corpus_index.load_from_jsonl(str(file_path))
            for file_path in self.corpus_dir.glob("*.json"):
                self.corpus_index.load_from_jsonl(str(file_path))
        else:
            try:
                self._load_from_hf_corpus()
            except Exception as e:
                if self.mode == "real":
                    raise RealDataUnavailableError(
                        f"FinanceBench real-mode corpus load failed: no corpus_path given, "
                        f"{self.corpus_dir} does not exist, and the Hugging Face corpus could not "
                        f"be loaded ({e}). Real benchmark execution must never silently use sample data."
                    ) from e
                print(f"Could not load HF corpus: {e}")
                self._create_sample_corpus()

        return self.corpus_index

    def _load_from_hf_corpus(self):
        from datasets import load_dataset

        ds = load_dataset("PatronusAI/financebench", "corpus", split="train")

        for item in ds:
            doc_id = item.get("id", item.get("doc_id", ""))
            text = item.get("text", item.get("content", ""))
            if doc_id and text:
                self.corpus_index.add_document(to_corpus_document({"id": doc_id, "text": text}))

    def _create_sample_corpus(self):
        sample_docs = [
            {
                "id": "apple_10k_2022.pdf",
                "text": "Apple Inc. 2022 Form 10-K. Total net sales for fiscal year 2022 were $394.33 billion, an increase of 8% from $365.82 billion in 2021. iPhone revenue was $205.49 billion. Services revenue was $78.13 billion. Gross margin was 43.3%."
            },
            {
                "id": "msft_10k_2023.pdf",
                "text": "Microsoft Corporation 2023 Form 10-K. Revenue was $211.92 billion. Gross margin was 68.9%. Operating income was $88.52 billion. Total debt was $47.03 billion. Total equity was $104.75 billion. Debt-to-equity ratio: 0.45."
            },
            {
                "id": "amzn_10k_2022.pdf",
                "text": "Amazon.com Inc. 2022 Form 10-K. Total revenue was $513.98 billion. Research and development expenses were $73.2 billion. AWS revenue was $80.1 billion. Operating income was $12.25 billion."
            },
            {
                "id": "apple_10k_2023.pdf",
                "text": "Apple Inc. 2023 Form 10-K. Total net sales for fiscal year 2023 were $383.28 billion. Total debt was $110.08 billion. Total equity was $63.65 billion. Debt-to-equity ratio: 1.73."
            },
            {
                "id": "tsla_10q_q4_2023.pdf",
                "text": "Tesla Inc. Q4 2023 Form 10-Q. Free cash flow for Q4 2023 was $2.06 billion. Total revenue was $25.17 billion. Automotive revenue was $21.56 billion. Operating cash flow was $3.49 billion. Capital expenditures were $1.43 billion."
            }
        ]

        for doc in sample_docs:
            self.corpus_index.add_document(to_corpus_document(doc))

    def run_evaluation(self, harness: DeepResearchHarness, max_examples: int = 20) -> Dict[str, Any]:
        if not self.examples:
            self.load_data(max_examples)

        if self.corpus_index is None:
            self.load_corpus()

        results = []

        for example in self.examples[:max_examples]:
            print(f"\nEvaluating: {example.question_id} ({example.question_type})")

            if example.question_type == "reasoning":
                criteria = [SufficiencyCriterion.CONSTRAINT_SATISFIED, SufficiencyCriterion.EVIDENCE_THRESHOLD]
                min_chunks = 3
            elif example.question_type == "calculation":
                criteria = [SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.EVIDENCE_THRESHOLD]
                min_chunks = 2
            else:
                criteria = [SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.EVIDENCE_THRESHOLD]
                min_chunks = 1

            harness.config.sufficiency_config = SufficiencyConfig(
                criteria=criteria,
                min_evidence_chunks=min_chunks
            )
            harness.sufficiency_checker = type(harness.sufficiency_checker)(harness.config.sufficiency_config)

            trajectory = harness.run_trajectory(
                query=example.question,
                query_id=example.question_id,
                gold_chunk_ids=example.gold_chunk_ids,
                benchmark="financebench"
            )

            predicted_answer = self._extract_answer(trajectory)

            match_score = self._score_answer(predicted_answer, example.answer, example.question_type)

            result = {
                "question_id": example.question_id,
                "question": example.question,
                "question_type": example.question_type,
                "doc_id": example.doc_id,
                "gold_answer": example.answer,
                "predicted_answer": predicted_answer,
                "match_score": match_score,
                "trajectory_recall": trajectory.trajectory_recall,
                "output_recall": trajectory.output_recall,
                "sufficiency_decision": trajectory.sufficiency_decision,
                "final_answer": trajectory.final_answer
            }
            results.append(result)

        return self._aggregate_results(results)

    def _extract_answer(self, trajectory: Trajectory) -> str:
        answer = trajectory.final_answer
        import re
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        first_line = lines[0] if lines else answer
        first_line = re.sub(r'\[.*?\]', '', first_line).strip()
        return first_line

    def _score_answer(self, predicted: str, gold: str, q_type: str) -> float:
        import re

        pred_norm = self._normalize(predicted)
        gold_norm = self._normalize(gold)

        if q_type in ["extraction", "calculation"]:
            pred_nums = re.findall(r'[\d,]+\.?\d*', pred_norm)
            gold_nums = re.findall(r'[\d,]+\.?\d*', gold_norm)

            if pred_nums and gold_nums:
                try:
                    pred_val = float(pred_nums[0].replace(',', ''))
                    gold_val = float(gold_nums[0].replace(',', ''))
                    if gold_val != 0:
                        rel_error = abs(pred_val - gold_val) / abs(gold_val)
                        return 1.0 if rel_error < 0.05 else 0.0
                except:
                    pass

        return 1.0 if pred_norm == gold_norm else 0.0

    def _normalize(self, text: str) -> str:
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[$,%]', '', text)
        text = re.sub(r'[.,;:]+$', '', text)
        return text

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        total = len(results)

        exact_matches = sum(1 for r in results if r["match_score"] > 0.5)

        by_type = {}
        for r in results:
            q_type = r.get("question_type", "unknown")
            if q_type not in by_type:
                by_type[q_type] = {"total": 0, "correct": 0}
            by_type[q_type]["total"] += 1
            if r["match_score"] > 0.5:
                by_type[q_type]["correct"] += 1

        type_accuracy = {t: v["correct"]/v["total"] for t, v in by_type.items()}

        return {
            "benchmark": "financebench",
            "total_examples": total,
            "overall_accuracy": exact_matches / total,
            "accuracy_by_type": type_accuracy,
            "avg_trajectory_recall": sum(r["trajectory_recall"] for r in results) / total,
            "avg_output_recall": sum(r["output_recall"] for r in results) / total,
            "avg_sufficiency_accuracy": sum(1 for r in results if r["sufficiency_decision"]) / total,
            "per_example": results
        }


def create_financebench_benchmark(data_dir: str = None, corpus_dir: str = None, mode: str = "test") -> FinanceBenchBenchmark:
    return FinanceBenchBenchmark(data_dir, corpus_dir, mode=mode)