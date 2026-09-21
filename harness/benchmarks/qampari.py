import json
import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import re
from ..core.models import Trajectory, Constraint
from ..core.harness import DeepResearchHarness, HarnessConfig
from ..core.metrics import RecallMetrics
from ..core.sufficiency_check import SufficiencyConfig, SufficiencyCriterion


@dataclass
class QampariExample:
    question_id: str
    question: str
    answers: List[str]
    answer_paragraphs: List[str]
    gold_chunk_ids: List[str] = field(default_factory=list)


class QampariExhaustivenessScorer:

    def __init__(self, fuzzy_match: bool = True, case_sensitive: bool = False):
        self.fuzzy_match = fuzzy_match
        self.case_sensitive = case_sensitive

    def normalize_answer(self, answer: str) -> str:
        if not self.case_sensitive:
            answer = answer.lower()
        answer = re.sub(r'\s+', ' ', answer.strip())
        answer = re.sub(r'^[.,;:]+|[.,;:]+$', '', answer)
        return answer

    def score(self, predicted_answers: List[str], gold_answers: List[str]) -> Dict[str, Any]:
        pred_norm = [self.normalize_answer(a) for a in predicted_answers]
        gold_norm = [self.normalize_answer(a) for a in gold_answers]

        pred_set = set(pred_norm)
        gold_set = set(gold_norm)

        exact_matches = pred_set & gold_set

        fuzzy_matches = set()
        if self.fuzzy_match:
            for p in pred_set:
                for g in gold_set:
                    if p in g or g in p:
                        fuzzy_matches.add(p)
                        break

        all_matches = exact_matches | fuzzy_matches

        precision = len(all_matches) / len(pred_set) if pred_set else 0.0
        recall = len(all_matches) / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        exhaustiveness = 1.0 if gold_set.issubset(all_matches) else 0.0

        missing = gold_set - all_matches
        extra = pred_set - all_matches

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exhaustiveness": exhaustiveness,
            "exact_match": 1.0 if pred_set == gold_set else 0.0,
            "num_predicted": len(pred_set),
            "num_gold": len(gold_set),
            "num_correct": len(all_matches),
            "correct_answers": list(all_matches),
            "missing_answers": list(missing),
            "extra_answers": list(extra),
            "gold_answers": list(gold_set),
            "predicted_answers": list(pred_set)
        }


class QampariBenchmark:

    def __init__(self, data_path: str = None, corpus_path: str = None):
        self.data_path = data_path
        self.corpus_path = corpus_path
        self.examples: List[QampariExample] = []
        self.corpus_index = None
        self.scorer = QampariExhaustivenessScorer()

    def load_data(self, max_examples: int = None) -> List[QampariExample]:
        if self.data_path and os.path.exists(self.data_path):
            return self._load_from_file(self.data_path, max_examples)

        try:
            return self._load_from_hf(max_examples)
        except Exception as e:
            print(f"Could not load from HF: {e}")

        print("Using sample QAMPARI data for testing")
        return self._create_sample_data(max_examples or 20)

    def _load_from_file(self, path: str, max_examples: int = None) -> List[QampariExample]:
        examples = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                data = json.loads(line)
                ex = QampariExample(
                    question_id=data.get("question_id", f"q_{i}"),
                    question=data.get("question", ""),
                    answers=data.get("answers", []),
                    answer_paragraphs=data.get("answer_paragraphs", []),
                    gold_chunk_ids=data.get("gold_chunk_ids", [])
                )
                examples.append(ex)
        self.examples = examples
        return examples

    def _load_from_hf(self, max_examples: int = None) -> List[QampariExample]:
        from datasets import load_dataset

        dataset_names = ["samsam3232/qampari", "qampari", "prime-qa/qampari"]
        ds = None

        for name in dataset_names:
            try:
                ds = load_dataset(name, split="train")
                break
            except:
                continue

        if ds is None:
            raise ValueError("Could not load QAMPARI from any known HF dataset")

        examples = []
        for i, item in enumerate(ds):
            if max_examples and i >= max_examples:
                break

            question = item.get("question") or item.get("query") or ""
            answers = item.get("answers") or item.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]

            ex = QampariExample(
                question_id=item.get("question_id", item.get("id", f"q_{i}")),
                question=question,
                answers=answers,
                answer_paragraphs=item.get("answer_paragraphs", item.get("paragraphs", [])),
                gold_chunk_ids=item.get("gold_chunk_ids", [])
            )
            examples.append(ex)

        self.examples = examples
        return examples

    def _create_sample_data(self, num_examples: int) -> List[QampariExample]:
        sample_questions = [
            {
                "question_id": "qampari_1",
                "question": "Which American actors have won the Academy Award for Best Actor?",
                "answers": ["Daniel Day-Lewis", "Tom Hanks", "Denzel Washington", "Jack Nicholson", "Marlon Brando"],
                "answer_paragraphs": ["p1", "p2", "p3", "p4", "p5"]
            },
            {
                "question_id": "qampari_2",
                "question": "What are the countries that border France?",
                "answers": ["Belgium", "Germany", "Italy", "Spain", "Switzerland", "Luxembourg", "Andorra", "Monaco"],
                "answer_paragraphs": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            },
            {
                "question_id": "qampari_3",
                "question": "Which novels were written by Jane Austen?",
                "answers": ["Pride and Prejudice", "Sense and Sensibility", "Emma", "Mansfield Park", "Northanger Abbey", "Persuasion"],
                "answer_paragraphs": ["p1", "p2", "p3", "p4", "p5", "p6"]
            },
            {
                "question_id": "qampari_4",
                "question": "What are the programming languages created by Guido van Rossum?",
                "answers": ["Python"],
                "answer_paragraphs": ["p1"]
            },
            {
                "question_id": "qampari_5",
                "question": "Which cities have hosted the Summer Olympics?",
                "answers": ["Athens", "Paris", "London", "Tokyo", "Los Angeles", "Rio de Janeiro", "Sydney", "Beijing"],
                "answer_paragraphs": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            }
        ]

        examples = []
        for i in range(min(num_examples, len(sample_questions))):
            q = sample_questions[i]
            ex = QampariExample(
                question_id=q["question_id"],
                question=q["question"],
                answers=q["answers"],
                answer_paragraphs=q["answer_paragraphs"]
            )
            examples.append(ex)

        while len(examples) < num_examples:
            base = sample_questions[len(examples) % len(sample_questions)]
            ex = QampariExample(
                question_id=f"{base['question_id']}_v{len(examples)}",
                question=base["question"],
                answers=base["answers"],
                answer_paragraphs=base["answer_paragraphs"]
            )
            examples.append(ex)

        self.examples = examples
        return examples

    def load_corpus(self, corpus_path: str = None):
        from ..core.search_read import InMemoryCorpusIndex

        self.corpus_index = InMemoryCorpusIndex(chunk_size=512, chunk_overlap=50)

        if corpus_path and os.path.exists(corpus_path):
            self.corpus_index.load_from_jsonl(corpus_path)
        else:
            self._create_sample_corpus()

        return self.corpus_index

    def _create_sample_corpus(self):
        sample_docs = [
            {
                "id": "wiki_1",
                "text": "Daniel Day-Lewis is an English actor who has won three Academy Awards for Best Actor, for My Left Foot (1989), There Will Be Blood (2007), and Lincoln (2012). He is the only male actor to have won three Oscars in the lead acting category."
            },
            {
                "id": "wiki_2",
                "text": "Tom Hanks is an American actor and filmmaker who has won two Academy Awards for Best Actor, for Philadelphia (1993) and Forrest Gump (1994). He is one of the most popular and recognizable film stars worldwide."
            },
            {
                "id": "wiki_3",
                "text": "Denzel Washington is an American actor, director, and producer who has won two Academy Awards: Best Supporting Actor for Glory (1989) and Best Actor for Training Day (2001)."
            },
            {
                "id": "wiki_4",
                "text": "Jack Nicholson is an American actor and filmmaker who has won three Academy Awards: Best Actor for One Flew Over the Cuckoo's Nest (1975) and As Good as It Gets (1997), and Best Supporting Actor for Terms of Endearment (1983)."
            },
            {
                "id": "wiki_5",
                "text": "Marlon Brando was an American actor and film director who won two Academy Awards for Best Actor: On the Waterfront (1954) and The Godfather (1972). He is widely regarded as one of the greatest actors of all time."
            },
            {
                "id": "wiki_6",
                "text": "France is a country in Western Europe that shares borders with Belgium to the northeast, Germany to the east, Switzerland to the southeast, Italy to the south, Spain to the southwest, and Luxembourg to the north. It also borders the microstates of Andorra and Monaco."
            },
            {
                "id": "wiki_7",
                "text": "Jane Austen was an English novelist known primarily for her six major novels: Sense and Sensibility (1811), Pride and Prejudice (1813), Mansfield Park (1814), Emma (1815), Northanger Abbey (1817), and Persuasion (1817)."
            },
            {
                "id": "wiki_8",
                "text": "Guido van Rossum is a Dutch programmer best known as the creator of the Python programming language, for which he was the 'benevolent dictator for life' (BDFL) until he stepped down from the position in July 2018."
            }
        ]

        for doc in sample_docs:
            self.corpus_index.add_document(doc)

    def run_evaluation(self, harness: DeepResearchHarness, max_examples: int = 20) -> Dict[str, Any]:
        if not self.examples:
            self.load_data(max_examples)

        if self.corpus_index is None:
            self.load_corpus()

        results = []

        for example in self.examples[:max_examples]:
            print(f"\nEvaluating: {example.question_id}")

            harness.config.sufficiency_config = SufficiencyConfig(
                criteria=[SufficiencyCriterion.ALL_ANSWERS, SufficiencyCriterion.EVIDENCE_THRESHOLD],
                expected_answer_count=len(example.answers),
                min_evidence_chunks=len(example.answers)
            )
            harness.sufficiency_checker = type(harness.sufficiency_checker)(harness.config.sufficiency_config)

            trajectory = harness.run_trajectory(
                query=example.question,
                query_id=example.question_id,
                gold_chunk_ids=example.gold_chunk_ids,
                benchmark="qampari"
            )

            predicted_answers = self._extract_answers(trajectory)

            exhaustiveness_score = self.scorer.score(predicted_answers, example.answers)

            recall_metrics = RecallMetrics(
                trajectory_recall=trajectory.trajectory_recall,
                output_recall=trajectory.output_recall
            )

            result = {
                "question_id": example.question_id,
                "question": example.question,
                "gold_answers": example.answers,
                "predicted_answers": predicted_answers,
                "exhaustiveness": exhaustiveness_score,
                "trajectory_recall": trajectory.trajectory_recall,
                "output_recall": trajectory.output_recall,
                "sufficiency_decision": trajectory.sufficiency_decision,
                "sufficiency_reason": trajectory.sufficiency_reason,
                "final_answer": trajectory.final_answer
            }
            results.append(result)

        return self._aggregate_results(results)

    def _extract_answers(self, trajectory: Trajectory) -> List[str]:
        answer = trajectory.final_answer

        answers = []

        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\[.*?\]', '', line)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line and len(line) > 2:
                answers.append(line.strip())

        if len(answers) <= 1:
            parts = re.split(r'[;,]', answer)
            for part in parts:
                part = re.sub(r'\[.*?\]', '', part).strip()
                if part and len(part) > 2:
                    answers.append(part)

        seen = set()
        unique_answers = []
        for a in answers:
            norm = a.lower()
            if norm not in seen:
                seen.add(norm)
                unique_answers.append(a)

        return unique_answers

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        total = len(results)
        exact_match = sum(1 for r in results if r["exhaustiveness"]["exact_match"] > 0.5)
        exhaustiveness = sum(1 for r in results if r["exhaustiveness"]["exhaustiveness"] > 0.5)

        avg_precision = sum(r["exhaustiveness"]["precision"] for r in results) / total
        avg_recall = sum(r["exhaustiveness"]["recall"] for r in results) / total
        avg_f1 = sum(r["exhaustiveness"]["f1"] for r in results) / total
        avg_trajectory_recall = sum(r["trajectory_recall"] for r in results) / total
        avg_output_recall = sum(r["output_recall"] for r in results) / total
        avg_sufficiency = sum(1 for r in results if r["sufficiency_decision"]) / total

        return {
            "benchmark": "qampari",
            "total_examples": total,
            "exact_match_accuracy": exact_match / total,
            "exhaustiveness_accuracy": exhaustiveness / total,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_trajectory_recall": avg_trajectory_recall,
            "avg_output_recall": avg_output_recall,
            "avg_sufficiency_accuracy": avg_sufficiency,
            "per_example": results
        }


def create_qampari_benchmark(data_path: str = None, corpus_path: str = None) -> QampariBenchmark:
    return QampariBenchmark(data_path, corpus_path)