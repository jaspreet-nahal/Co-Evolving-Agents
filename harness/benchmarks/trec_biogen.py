import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrecBiogenExample:
    query_id: str
    query: str
    gold_doc_ids: List[str] = field(default_factory=list)
    narrative: str = ""


class TrecBiogenBenchmark:

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/trec_biogen")
        self.examples: List[TrecBiogenExample] = []
        self.corpus_index = None
        self._verified = False

    def verify_data_access(self) -> Dict[str, Any]:
        results = {
            "track_found": False,
            "data_access_method": None,
            "license_terms": None,
            "requires_agreement": None,
            "corpus_available": False,
            "topics_available": False,
            "qrels_available": False,
            "notes": []
        }

        results["notes"].append("VERIFICATION PENDING - Need to check trec.nist.gov and track website")
        results["notes"].append("Expected: TREC 2024 Biomedical Generative Retrieval track")
        results["notes"].append("Likely requires NIST data-use agreement")

        return results

    def load_data(self, max_examples: int = None) -> List[TrecBiogenExample]:
        if not self._verified:
            verification = self.verify_data_access()
            print("DATA ACCESS VERIFICATION REQUIRED:")
            print(f"  {verification}")
            raise RuntimeError("TREC-Biogen data access not verified. Run verify_data_access() first.")

        self.examples = []
        return []

    def load_corpus(self):
        if not self._verified:
            raise RuntimeError("Data access not verified")

        return None

    def run_evaluation(self, harness, max_examples: int = 20) -> Dict[str, Any]:
        if not self._verified:
            raise RuntimeError("Data access not verified")

        return {
            "benchmark": "trec_biogen",
            "status": "not_implemented",
            "reason": "Data access verification required"
        }


def create_trec_biogen_benchmark(data_dir: str = None) -> TrecBiogenBenchmark:
    return TrecBiogenBenchmark(data_dir)


VERIFICATION_CHECKLIST = """
TREC-Biogen 2024 Verification Checklist:
  1. Check https://trec.nist.gov/ for 2024 tracks
  2. Search for "TREC 2024 BioGen" or "Biomedical Generative Retrieval"
  3. Find track website (often hosted by organizing institution)
  4. Check data access requirements:
      - Data-use agreement required?
      - Academic affiliation required?
      - Registration deadline?
  5. Confirm available data:
      - Topics (queries with narratives)
      - Corpus (PubMed Central, etc.)
      - Qrels (relevance judgments)
  6. Confirm evaluation metrics (nDCG@10, MAP, etc.)
  7. Download and verify data format
  8. Report findings to proceed with implementation
"""