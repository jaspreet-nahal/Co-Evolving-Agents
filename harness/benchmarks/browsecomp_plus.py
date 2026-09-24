import os
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from ..core.models import Trajectory
from ..core.harness import DeepResearchHarness, HarnessConfig
from ..core.metrics import RecallMetrics
from ..core.sufficiency_check import SufficiencyConfig, SufficiencyCriterion
from ..core.search_read import to_corpus_document

@dataclass
class BrowseCompExample:
    question_id: str
    question: str
    answer: str
    gold_chunk_ids: List[str] = field(default_factory=list)


class RealDataUnavailableError(RuntimeError):
    """Raised in mode='real' when a verified real corpus/dataset cannot be
    loaded. Real benchmark execution must never silently substitute sample
    data; callers must fix the data path/HF access or fall back to
    mode='test' explicitly, never implicitly."""


class BrowseCompPlusBenchmark:

    def __init__(self, data_dir: str = None, corpus_dir: str = None, index_dir: str = None, mode: str = "test"):
        if mode not in ("test", "real"):
            raise ValueError(f"mode must be 'test' or 'real', got {mode!r}")
        self.data_dir = Path(data_dir) if data_dir else Path("data/browsecomp_plus")
        self.corpus_dir = Path(corpus_dir) if corpus_dir else Path("data/browsecomp_plus_corpus")
        self.index_dir = Path(index_dir) if index_dir else Path("data/browsecomp_plus_indexes")
        self.mode = mode
        self.examples: List[BrowseCompExample] = []
        self.corpus_index = None

    def decrypt_dataset(self, encrypted_path: str, output_path: str) -> bool:

        decrypt_script = self.data_dir / "scripts_build_index" / "decrypt_dataset.py"
        if not decrypt_script.exists():
            print("Cloning BrowseComp-Plus repo for decrypt script...")
            try:
                subprocess.run([
                    "git", "clone", "https://github.com/texttron/BrowseComp-Plus.git",
                    str(self.data_dir / "BrowseComp-Plus")
                ], check=True, capture_output=True)
                decrypt_script = self.data_dir / "BrowseComp-Plus" / "scripts_build_index" / "decrypt_dataset.py"
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repo: {e}")
                return False

        try:
            result = subprocess.run([
                sys.executable, str(decrypt_script),
                "--input", encrypted_path,
                "--output", output_path
            ], check=True, capture_output=True, text=True)
            print(f"Decryption successful: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Decryption failed: {e.stderr}")
            return False

    def load_data(self, max_examples: int = None, decrypted_path: str = None) -> List[BrowseCompExample]:
        if decrypted_path and os.path.exists(decrypted_path):
            return self._load_from_decrypted(decrypted_path, max_examples)

        try:
            return self._load_from_hf_encrypted(max_examples)
        except Exception as e:
            if self.mode == "real":
                raise RealDataUnavailableError(
                    f"BrowseComp-Plus real-mode data load failed: no decrypted_path given/found and "
                    f"the encrypted Hugging Face dataset could not be loaded/decrypted ({e}). Real "
                    f"benchmark execution must never silently use sample data; fix the data source "
                    f"or use mode='test'."
                ) from e
            print(f"Could not load from HF: {e}")

        print("Using sample BrowseComp-Plus data for testing (mode='test')")
        return self._create_sample_data(max_examples or 20)

    def _load_from_decrypted(self, path: str, max_examples: int = None) -> List[BrowseCompExample]:
        examples = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                data = json.loads(line)
                ex = BrowseCompExample(
                    question_id=data.get("id", data.get("question_id", f"bc_{i}")),
                    question=data.get("question", data.get("query", "")),
                    answer=data.get("answer", data.get("gold_answer", "")),
                    gold_chunk_ids=data.get("gold_chunk_ids", data.get("evidence_docs", []))
                )
                examples.append(ex)
        self.examples = examples
        return examples

    def _load_from_hf_encrypted(self, max_examples: int = None) -> List[BrowseCompExample]:
        from datasets import load_dataset

        ds = load_dataset("Tevatron/browsecomp-plus", split="train")

        encrypted_path = self.data_dir / "encrypted_queries.jsonl"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(encrypted_path, 'w') as f:
            for item in ds:
                f.write(json.dumps(item) + "\n")

        decrypted_path = self.data_dir / "decrypted_queries.jsonl"
        if not self.decrypt_dataset(str(encrypted_path), str(decrypted_path)):
            raise RuntimeError("Failed to decrypt dataset")

        return self._load_from_decrypted(str(decrypted_path), max_examples)

    def _create_sample_data(self, num_examples: int) -> List[BrowseCompExample]:
        sample_questions = [
            {
                "question_id": "bc_1",
                "question": "What is the name of the researcher who developed the transformer architecture in the paper 'Attention Is All You Need'?",
                "answer": "Ashish Vaswani",
                "gold_chunk_ids": ["doc_1_chunk_0", "doc_1_chunk_1"]
            },
            {
                "question_id": "bc_2",
                "question": "Which company acquired DeepMind in 2014?",
                "answer": "Google",
                "gold_chunk_ids": ["doc_2_chunk_0"]
            },
            {
                "question_id": "bc_3",
                "question": "What is the capital of the country that hosted the 2022 FIFA World Cup?",
                "answer": "Doha",
                "gold_chunk_ids": ["doc_3_chunk_0", "doc_3_chunk_1"]
            },
            {
                "question_id": "bc_4",
                "question": "Who wrote the novel 'The Great Gatsby'?",
                "answer": "F. Scott Fitzgerald",
                "gold_chunk_ids": ["doc_4_chunk_0"]
            },
            {
                "question_id": "bc_5",
                "question": "What year was the first iPhone released?",
                "answer": "2007",
                "gold_chunk_ids": ["doc_5_chunk_0"]
            }
        ]

        examples = []
        for i in range(min(num_examples, len(sample_questions))):
            q = sample_questions[i]
            ex = BrowseCompExample(
                question_id=q["question_id"],
                question=q["question"],
                answer=q["answer"],
                gold_chunk_ids=q["gold_chunk_ids"]
            )
            examples.append(ex)

        while len(examples) < num_examples:
            base = sample_questions[len(examples) % len(sample_questions)]
            ex = BrowseCompExample(
                question_id=f"{base['question_id']}_v{len(examples)}",
                question=base["question"],
                answer=base["answer"],
                gold_chunk_ids=base["gold_chunk_ids"]
            )
            examples.append(ex)

        self.examples = examples
        return examples

    def load_corpus(self, corpus_path: str = None):
        from ..core.search_read import InMemoryCorpusIndex

        self.corpus_index = InMemoryCorpusIndex(chunk_size=512, chunk_overlap=50)

        if corpus_path:
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(
                    f"BrowseComp-Plus corpus_path '{corpus_path}' was requested but does not exist. "
                    f"Refusing to silently substitute sample data for an explicitly requested corpus."
                )
            from datasets import load_dataset
            ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")

            for item in ds:
                doc_id = item.get("id", item.get("doc_id", ""))
                text = item.get("text", item.get("content", ""))
                if doc_id and text:
                    self.corpus_index.add_document(to_corpus_document({"id": doc_id, "text": text}))

        elif self.index_dir.exists():
            raise NotImplementedError(
                f"Pre-built index directory '{self.index_dir}' exists but pre-built index loading is "
                f"not implemented. Refusing to silently substitute a 5-document sample corpus for what "
                f"was set up as a real pre-built index (see architecture.md limitation D-020/D-022)."
            )

        elif self.mode == "real":
            raise RealDataUnavailableError(
                "BrowseComp-Plus real-mode corpus load requires an explicit, verified corpus_path "
                "or pre-built index directory; neither was found. Real benchmark execution must "
                "never silently use sample data."
            )

        else:
            self._create_sample_corpus()

        return self.corpus_index

    def _create_sample_corpus(self):
        sample_docs = [
            {"id": "doc_1", "text": "The Transformer architecture was developed by researchers at Google. Ashish Vaswani was the lead author of the paper 'Attention Is All You Need' published in 2017. The paper introduced the self-attention mechanism that revolutionized NLP."},
            {"id": "doc_2", "text": "DeepMind is a British AI company founded in 2010. It was acquired by Google in 2014 for approximately $500 million. DeepMind is known for AlphaGo and AlphaFold."},
            {"id": "doc_3", "text": "The 2022 FIFA World Cup was hosted by Qatar. The capital of Qatar is Doha. The tournament took place from November to December 2022."},
            {"id": "doc_4", "text": "The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. Set in the Jazz Age, it tells the story of Jay Gatsby."},
            {"id": "doc_5", "text": "The first iPhone was announced by Steve Jobs on January 9, 2007 and released on June 29, 2007 in the United States. It revolutionized the smartphone industry."},
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
            print(f"\nEvaluating: {example.question_id}")

            harness.config.sufficiency_config = SufficiencyConfig(
                criteria=[SufficiencyCriterion.ANY_ANSWER, SufficiencyCriterion.CONSTRAINT_SATISFIED],
                min_evidence_chunks=2
            )
            harness.sufficiency_checker = type(harness.sufficiency_checker)(harness.config.sufficiency_config)

            trajectory = harness.run_trajectory(
                query=example.question,
                query_id=example.question_id,
                gold_chunk_ids=example.gold_chunk_ids,
                benchmark="browsecomp_plus"
            )

            predicted_answer = self._extract_answer(trajectory)

            exact_match = self._normalize(predicted_answer) == self._normalize(example.answer)

            result = {
                "question_id": example.question_id,
                "question": example.question,
                "gold_answer": example.answer,
                "predicted_answer": predicted_answer,
                "exact_match": exact_match,
                "trajectory_recall": trajectory.trajectory_recall,
                "output_recall": trajectory.output_recall,
                "sufficiency_decision": trajectory.sufficiency_decision,
                "final_answer": trajectory.final_answer
            }
            results.append(result)

        return self._aggregate_results(results)

    def _extract_answer(self, trajectory: Trajectory) -> str:
        answer = trajectory.final_answer

        lines = answer.split('\n')
        first_line = lines[0] if lines else answer

        import re
        first_line = re.sub(r'\[.*?\]', '', first_line).strip()
        return first_line

    def _normalize(self, text: str) -> str:
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.,;:]+$', '', text)
        return text

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        total = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])

        return {
            "benchmark": "browsecomp_plus",
            "total_examples": total,
            "exact_match_accuracy": exact_matches / total,
            "avg_trajectory_recall": sum(r["trajectory_recall"] for r in results) / total,
            "avg_output_recall": sum(r["output_recall"] for r in results) / total,
            "avg_sufficiency_accuracy": sum(1 for r in results if r["sufficiency_decision"]) / total,
            "per_example": results
        }


def create_browsecomp_plus_benchmark(data_dir: str = None, corpus_dir: str = None, index_dir: str = None, mode: str = "test") -> BrowseCompPlusBenchmark:
    return BrowseCompPlusBenchmark(data_dir, corpus_dir, index_dir, mode=mode)