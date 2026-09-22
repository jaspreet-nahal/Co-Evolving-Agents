import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .models import Constraint

class ConstraintType(str, Enum):
    TEMPORAL = "temporal"
    ENTITY_EXCLUSION = "entity_exclusion"
    ENTITY_INCLUSION = "entity_inclusion"
    QUANTITY = "quantity"
    GEOGRAPHIC = "geographic"
    DOMAIN_SPECIFIC = "domain_specific"
    NEGATION = "negation"
    COMPARISON = "comparison"


@dataclass
class SubQuery:
    id: str
    text: str
    dependencies: List[str] = field(default_factory=list)
    expected_answer_type: str = "factual"
    priority: int = 1


@dataclass
class Plan:
    original_query: str
    constraints: List[Constraint] = field(default_factory=list)
    sub_queries: List[SubQuery] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class Planner:

    TEMPORAL_PATTERNS = [
        (r'\b(before|after|since|until|by|in|during)\s+(\d{4})\b', 'year'),
        (r'\b(before|after|since|until|by)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', 'date'),
        (r'\b(between|from)\s+(\d{4})\s+(and|to)\s+(\d{4})\b', 'year_range'),
        (r'\b(last|past|previous)\s+(\d+)\s+(years?|months?|days?)\b', 'relative_period'),
    ]

    EXCLUSION_PATTERNS = [
        (r'\b(excluding|except|without|not including)\s+([^.?,]+)', 'exclusion'),
        (r'\b(not|excluding)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'entity_exclusion'),
    ]

    INCLUSION_PATTERNS = [
        (r'\b(only|specifically|focusing on|limited to)\s+([^.?,]+)', 'inclusion'),
        (r'\b(including|such as)\s+([^.?,]+)', 'entity_inclusion'),
    ]

    QUANTITY_PATTERNS = [
        (r'\b(top|bottom|first|last)\s+(\d+)\b', 'ranking'),
        (r'\b(at least|at most|more than|less than|over|under)\s+(\d+)', 'threshold'),
        (r'\b(exactly|precisely)\s+(\d+)', 'exact_count'),
    ]

    def __init__(self, use_llm: bool = False, llm_client: Any = None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    def plan(self, query: str) -> Plan:
        if self.use_llm and self.llm_client:
            return self._plan_with_llm(query)
        else:
            return self._plan_rule_based(query)

    def _plan_rule_based(self, query: str) -> Plan:
        plan = Plan(original_query=query)

        plan.constraints = self._extract_constraints(query)

        plan.sub_queries = self._decompose_query(query, plan.constraints)

        plan.reasoning = self._generate_reasoning(plan)

        return plan

    def _extract_constraints(self, query: str) -> List[Constraint]:
        constraints = []
        query_lower = query.lower()

        for pattern, subtype in self.TEMPORAL_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                constraints.append(Constraint(
                    type=ConstraintType.TEMPORAL.value,
                    description=f"Temporal constraint: {match.group(0)}",
                    raw_text=match.group(0),
                    parsed_value={"subtype": subtype, "full_match": match.group(0), "groups": match.groups()}
                ))

        for pattern, subtype in self.EXCLUSION_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                constraints.append(Constraint(
                    type=ConstraintType.ENTITY_EXCLUSION.value,
                    description=f"Exclusion constraint: {match.group(0)}",
                    raw_text=match.group(0),
                    parsed_value={"subtype": subtype, "full_match": match.group(0), "groups": match.groups()}
                ))

        for pattern, subtype in self.INCLUSION_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                constraints.append(Constraint(
                    type=ConstraintType.ENTITY_INCLUSION.value,
                    description=f"Inclusion constraint: {match.group(0)}",
                    raw_text=match.group(0),
                    parsed_value={"subtype": subtype, "full_match": match.group(0), "groups": match.groups()}
                ))

        for pattern, subtype in self.QUANTITY_PATTERNS:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                constraints.append(Constraint(
                    type=ConstraintType.QUANTITY.value,
                    description=f"Quantity constraint: {match.group(0)}",
                    raw_text=match.group(0),
                    parsed_value={"subtype": subtype, "full_match": match.group(0), "groups": match.groups()}
                ))

        return constraints

    def _decompose_query(self, query: str, constraints: List[Constraint]) -> List[SubQuery]:
        sub_queries = []

        parts = re.split(r'\?|\band\b|\bor\b|,\s*', query)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

        if len(parts) <= 1:
            sub_queries.append(SubQuery(
                id="sq_1",
                text=query,
                expected_answer_type=self._infer_answer_type(query)
            ))
        else:
            for i, part in enumerate(parts):
                if part.endswith('?'):
                    sub_queries.append(SubQuery(
                        id=f"sq_{i+1}",
                        text=part,
                        expected_answer_type=self._infer_answer_type(part)
                    ))

        if not sub_queries:
            sub_queries.append(SubQuery(
                id="sq_1",
                text=query,
                expected_answer_type=self._infer_answer_type(query)
            ))

        return sub_queries

    def _infer_answer_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ['how many', 'count', 'number of']):
            return "count"
        elif any(w in query_lower for w in ['list', 'what are', 'which', 'enumerate']):
            return "list"
        elif any(w in query_lower for w in ['true', 'false', 'yes', 'no', 'is it', 'does']):
            return "boolean"
        elif any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs']):
            return "comparative"
        else:
            return "factual"

    def _generate_reasoning(self, plan: Plan) -> str:
        parts = [f"Original query: {plan.original_query}"]

        if plan.constraints:
            parts.append("\nExtracted constraints:")
            for c in plan.constraints:
                parts.append(f"  - [{c.type}] {c.description}")

        if plan.sub_queries:
            parts.append("\nDecomposed sub-queries:")
            for sq in plan.sub_queries:
                parts.append(f"  - {sq.id}: {sq.text} (type: {sq.expected_answer_type})")

        return "\n".join(parts)

    def _plan_with_llm(self, query: str) -> Plan:
        return self._plan_rule_based(query)


def create_planner(use_llm: bool = False, llm_client: Any = None) -> Planner:
    return Planner(use_llm=use_llm, llm_client=llm_client)