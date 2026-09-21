from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .models import SufficiencyResult, Constraint, Trajectory
from .working_memory import WorkingMemory, EvidenceChunk


class SufficiencyCriterion(str, Enum):
    ANY_ANSWER = "any_answer"
    ALL_ANSWERS = "all_answers"
    CONSTRAINT_SATISFIED = "constraint_satisfied"
    EVIDENCE_THRESHOLD = "evidence_threshold"
    CONFIDENCE_THRESHOLD = "confidence_threshold"


@dataclass
class SufficiencyConfig:
    criteria: List[SufficiencyCriterion] = field(default_factory=lambda: [SufficiencyCriterion.ANY_ANSWER])
    min_evidence_chunks: int = 3
    min_confidence: float = 0.7
    expected_answer_count: Optional[int] = None
    check_constraints: bool = True


class SufficiencyChecker:

    def __init__(self, config: SufficiencyConfig = None):
        self.config = config or SufficiencyConfig()

    def check(self, working_memory: WorkingMemory, trajectory: Trajectory,
              plan_constraints: List[Constraint] = None) -> SufficiencyResult:
        plan_constraints = plan_constraints or trajectory.constraints
        active_chunks = working_memory.get_active_chunks()

        reasons = []
        missing_info = []
        is_sufficient = True
        confidence = 0.0

        if SufficiencyCriterion.EVIDENCE_THRESHOLD in self.config.criteria:
            if len(active_chunks) < self.config.min_evidence_chunks:
                is_sufficient = False
                reasons.append(f"Insufficient evidence chunks: {len(active_chunks)} < {self.config.min_evidence_chunks}")
                missing_info.append(f"Need at least {self.config.min_evidence_chunks} evidence chunks")
            else:
                reasons.append(f"Evidence threshold met: {len(active_chunks)} chunks")

        if SufficiencyCriterion.CONSTRAINT_SATISFIED in self.config.criteria and self.config.check_constraints:
            constraint_check = self._check_constraints(active_chunks, plan_constraints)
            if not constraint_check["satisfied"]:
                is_sufficient = False
                reasons.append(f"Constraints not satisfied: {constraint_check['unsatisfied']}")
                missing_info.extend(constraint_check["missing"])
            else:
                reasons.append("All constraints satisfied")

        if SufficiencyCriterion.ALL_ANSWERS in self.config.criteria:
            exhaustiveness_check = self._check_exhaustiveness(active_chunks, trajectory)
            if not exhaustiveness_check["complete"]:
                is_sufficient = False
                reasons.append(f"Answer set incomplete: {exhaustiveness_check['found']}/{exhaustiveness_check['expected']} answers found")
                missing_info.append(f"Missing {exhaustiveness_check['expected'] - exhaustiveness_check['found']} answers")
            else:
                reasons.append(f"Complete answer set found: {exhaustiveness_check['found']} answers")

        if SufficiencyCriterion.ANY_ANSWER in self.config.criteria:
            if len(active_chunks) == 0:
                is_sufficient = False
                reasons.append("No evidence chunks retrieved")
                missing_info.append("Need at least one evidence chunk")
            else:
                reasons.append("At least one evidence chunk found")

        confidence = self._compute_confidence(active_chunks, reasons, is_sufficient)

        if SufficiencyCriterion.CONFIDENCE_THRESHOLD in self.config.criteria:
            if confidence < self.config.min_confidence:
                is_sufficient = False
                reasons.append(f"Confidence too low: {confidence:.2f} < {self.config.min_confidence}")
                missing_info.append(f"Need confidence >= {self.config.min_confidence}")

        reason_str = "; ".join(reasons) if reasons else "No criteria evaluated"

        return SufficiencyResult(
            is_sufficient=is_sufficient,
            reason=reason_str,
            missing_info=missing_info,
            confidence=confidence
        )

    def _check_constraints(self, active_chunks: List[EvidenceChunk],
                           constraints: List[Constraint]) -> Dict[str, Any]:
        unsatisfied = []
        missing = []

        for constraint in constraints:
            constraint_terms = self._extract_constraint_terms(constraint)
            found = False

            for chunk in active_chunks:
                chunk_text = chunk.chunk.content.lower()
                if any(term.lower() in chunk_text for term in constraint_terms):
                    found = True
                    break

            if not found:
                unsatisfied.append(constraint.description)
                missing.append(f"Evidence for constraint: {constraint.description}")

        return {
            "satisfied": len(unsatisfied) == 0,
            "unsatisfied": unsatisfied,
            "missing": missing
        }

    def _extract_constraint_terms(self, constraint: Constraint) -> List[str]:
        if constraint.parsed_value and "groups" in constraint.parsed_value:
            return [g for g in constraint.parsed_value["groups"] if g]
        return constraint.raw_text.split()

    def _check_exhaustiveness(self, active_chunks: List[EvidenceChunk],
                              trajectory: Trajectory) -> Dict[str, Any]:
        expected = self.config.expected_answer_count or 0

        found = min(len(active_chunks), expected) if expected > 0 else len(active_chunks)

        return {
            "complete": expected > 0 and found >= expected,
            "found": found,
            "expected": expected
        }

    def _compute_confidence(self, active_chunks: List[EvidenceChunk],
                            reasons: List[str], is_sufficient: bool) -> float:
        if not active_chunks:
            return 0.0

        chunk_confidence = min(len(active_chunks) / 10.0, 1.0)

        constraint_boost = 0.2 if "All constraints satisfied" in " ".join(reasons) else 0.0

        exhaustiveness_boost = 0.3 if "Complete answer set found" in " ".join(reasons) else 0.0

        confidence = min(chunk_confidence + constraint_boost + exhaustiveness_boost, 1.0)

        return confidence


def create_sufficiency_checker(config: SufficiencyConfig = None) -> SufficiencyChecker:
    return SufficiencyChecker(config)