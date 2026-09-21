import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class FreshStackExample:
    example_id: str
    question: str
    answer: str
    tech_stack: str = ""
    doc_type: str = ""
    gold_chunk_ids: List[str] = field(default_factory=list)
    procedural_steps: List[str] = field(default_factory=list)


class FreshStackBenchmark:
    def __init__(self, data_dir: str = None, split: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/freshstack")
        self.split = split
        self.examples: List[FreshStackExample] = []
        self.corpus_index = None
        self._verified = False

    def verify_data_access(self) -> Dict[str, Any]:
        results = {
            "org_accessible": False,
            "available_splits": [],
            "procedural_reasoning_splits": [],
            "selected_split": self.split,
            "data_format": None,
            "license": None,
            "notes": []
        }

        results["notes"].append("VERIFICATION PENDING - Need to check HF org and project page")
        results["notes"].append("Expected: Multiple splits by tech stack / doc type")
        results["notes"].append("Target: Procedural reasoning over technical documentation")

        return results

    def list_available_splits(self) -> List[str]:
        return []

    def load_data(self, max_examples: int = None, split: str = None) -> List[FreshStackExample]:
        if not self._verified and not split:
            verification = self.verify_data_access()
            print("DATA ACCESS VERIFICATION REQUIRED:")
            print(f"  {verification}")
            raise RuntimeError("FreshStack split not identified. Run verify_data_access() first.")

        target_split = split or self.split
        if not target_split:
            raise ValueError("No split specified. Must identify correct split first.")

        self.examples = []
        return []

    def load_corpus(self, split: str = None):
        if not self._verified:
            raise RuntimeError("Data access not verified")

        return None

    def run_evaluation(self, harness, max_examples: int = 20) -> Dict[str, Any]:
        if not self._verified:
            raise RuntimeError("Data access not verified")

        return {
            "benchmark": "freshstack",
            "status": "not_implemented",
            "reason": "Split identification and verification required"
        }


def create_freshstack_benchmark(data_dir: str = None, split: str = None) -> FreshStackBenchmark:
    return FreshStackBenchmark(data_dir, split)


VERIFICATION_CHECKLIST = """
FreshStack Verification Checklist:
  1. Check https://huggingface.co/freshstack - list all datasets
  2. Check https://fresh-stack.github.io - project documentation
  3. Identify dataset splits (likely by tech stack: python, javascript, rust, k8s, etc.)
  4. For each split, check:
      - Task type: Is it "procedural reasoning over docs"?
      - Data format: Questions, answers, supporting documents?
      - Size: How many examples?
  5. Select split(s) matching "procedural reasoning over docs"
  6. Confirm with team which split(s) to use
  7. Download sample and verify format
  8. Report findings to proceed with implementation

Expected relevant splits (hypotheses):
- freshstack/procedural-reasoning
- freshstack/technical-reasoning
- freshstack/{lang}/procedural (e.g., freshstack/python/procedural)
- freshstack/api-docs-qa
"""