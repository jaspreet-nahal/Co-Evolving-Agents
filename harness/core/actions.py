import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ActionType(str, Enum):
    FAN_OUT_SEARCH = "fan_out_search"
    SEARCH_CORPUS = "search_corpus"
    GREP_CORPUS = "grep_corpus"
    READ_DOCUMENT = "read_document"
    REVIEW_DOCS = "review_docs"
    CURATE = "curate"
    VERIFY = "verify"
    END_SEARCH = "end_search"


@dataclass
class HarnessAction:
    action: ActionType
    arguments: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "HarnessAction":
        if not isinstance(value, dict):
            raise ValueError("Action must be a JSON object")
        name = value.get("action", value.get("tool"))
        if name not in {item.value for item in ActionType}:
            raise ValueError(f"Unknown action: {name}")
        arguments = value.get("arguments", value.get("params", {}))
        if not isinstance(arguments, dict):
            raise ValueError("Action arguments must be an object")
        return cls(ActionType(name), arguments, str(value.get("reasoning", "")))

    @classmethod
    def from_json(cls, value: str) -> "HarnessAction":
        return cls.from_dict(json.loads(value))

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action.value, "arguments": self.arguments, "reasoning": self.reasoning}


def action_schema() -> List[Dict[str, Any]]:
    return [{"name": item.value, "arguments": "object"} for item in ActionType]