"""Canonical model-facing action contract and model-backed policies.

The contract is a single JSON object:

    {"action": "search_corpus", "arguments": {"query": "..."}, "reasoning": "..."}

`action` must be one of the ActionType values legal for the current
execution mode (H0's flat contract is a strict subset of H1's). Malformed
output (invalid JSON, unknown action, wrong shape) is never silently
reinterpreted or converted into a RuleBasedPolicy/RuleBasedFlatPolicy
decision: it raises MalformedModelOutputError, which the caller logs as a
model/policy error on the trajectory. A single deterministic re-prompt
retry is supported (`max_parse_retries`), and every attempt -- including
failures -- is logged.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .actions import ActionType, HarnessAction
from .model_backend import TransformersModelBackend


# H0's allowed action set (mirrors flat_react.FLAT_REACT_ALLOWED_ACTIONS,
# duplicated here as plain strings so this module does not need to import
# flat_react, avoiding a circular import; kept in sync by the shared
# ActionType enum and covered by a cross-check test).
H0_ALLOWED_ACTIONS = {ActionType.SEARCH_CORPUS, ActionType.GREP_CORPUS, ActionType.READ_DOCUMENT, ActionType.END_SEARCH}

# H1's allowed action set: every registered ActionType. Any future action
# added to ActionType is automatically legal for H1 and must be explicitly
# added to H0_ALLOWED_ACTIONS above to also be legal there.
H1_ALLOWED_ACTIONS = set(ActionType)


class MalformedModelOutputError(RuntimeError):
    """Raised when model output cannot be parsed into a legal HarnessAction.

    Never caught to silently substitute a rule-based decision. Callers
    must log this as a model/policy error (see ParseAttempt/attempts on
    ModelBackedPolicyBase) and let it propagate to the trajectory's error
    handling.
    """


@dataclass
class ParseAttempt:
    raw_text: str
    error: Optional[str] = None
    succeeded: bool = False


def extract_json_object(text: str) -> str:
    """Best-effort, deterministic extraction of a single JSON object from
    raw model text (e.g. if the model wraps it in markdown fences or adds
    leading/trailing prose). Does not alter the JSON content itself -- only
    trims surrounding non-JSON text. Returns the trimmed candidate string;
    it is the caller's job to json.loads it and handle failure.
    """
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return stripped[brace_start:brace_end + 1]
    return stripped


def parse_agent_action(text: str, allowed_actions: Optional[set] = None) -> HarnessAction:
    """Parse raw model text into a HarnessAction under the canonical contract.

    Raises MalformedModelOutputError (never returns a fallback action) if
    the text is not valid JSON, is missing required fields, or names an
    action outside `allowed_actions` (when given).
    """
    candidate = extract_json_object(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise MalformedModelOutputError(f"Model output is not valid JSON: {error}. Raw text: {text!r}") from error

    try:
        action = HarnessAction.from_dict(value)
    except ValueError as error:
        raise MalformedModelOutputError(f"Model output failed the action contract: {error}. Raw text: {text!r}") from error

    if allowed_actions is not None and action.action not in allowed_actions:
        raise MalformedModelOutputError(
            f"Action '{action.action.value}' is not permitted in this execution mode "
            f"(allowed: {sorted(a.value for a in allowed_actions)}). Raw text: {text!r}"
        )
    return action


class ModelBackedPolicyBase:
    """Shared machinery for the H0 and H1 model-backed policies.

    Holds a bounded, deterministic retry: if the model's first response
    fails to parse, it is re-prompted once with the parse error appended
    (never with a different action set or a weaker instruction), and if
    that also fails, MalformedModelOutputError propagates. Every attempt
    is recorded in `self.attempts` for logging.
    """

    system_prompt = (
        "You are a research agent. On every turn, respond with exactly one JSON object "
        "and nothing else: {\"action\": <action_name>, \"arguments\": {...}, \"reasoning\": \"...\"}. "
        "Do not include any text outside the JSON object."
    )

    def __init__(self, backend: TransformersModelBackend, allowed_actions: set, max_parse_retries: int = 1):
        self.backend = backend
        self.allowed_actions = allowed_actions
        self.max_parse_retries = max_parse_retries
        self.attempts: List[ParseAttempt] = []

    def _action_list_text(self) -> str:
        return ", ".join(sorted(a.value for a in self.allowed_actions))

    def _generate_and_parse(self, prompt: str) -> HarnessAction:
        last_error: Optional[Exception] = None
        current_prompt = prompt
        for attempt_index in range(self.max_parse_retries + 1):
            raw_text = self.backend.generate(current_prompt)
            try:
                action = parse_agent_action(raw_text, self.allowed_actions)
                self.attempts.append(ParseAttempt(raw_text=raw_text, succeeded=True))
                return action
            except MalformedModelOutputError as error:
                self.attempts.append(ParseAttempt(raw_text=raw_text, error=str(error), succeeded=False))
                last_error = error
                current_prompt = (
                    f"{prompt}\n\nYour previous response could not be parsed: {error}\n"
                    f"Respond again with exactly one valid JSON object as instructed."
                )
        raise last_error


class ModelBackedFlatPolicy(ModelBackedPolicyBase):
    """H0 (flat) action policy backed by a real frozen model.

    Receives FlatReActExecutor's `_FlatPolicyView` (query, turn, budget,
    raw transcript) -- structurally the same information a plain ReAct
    loop would show the model, nothing more. Prompt content and generation
    settings are identical in spirit to ModelBackedEpisodePolicy (Part G):
    the only difference is which action set/state view is legal, which is
    the harness intervention itself, not an extra capability given to one
    condition.
    """

    def __init__(self, backend: TransformersModelBackend, max_parse_retries: int = 1):
        super().__init__(backend, H0_ALLOWED_ACTIONS, max_parse_retries)

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        prompt = (
            f"{self.system_prompt}\n"
            f"Valid actions: {self._action_list_text()}.\n"
            f"{observation}\n"
        )
        return self._generate_and_parse(prompt)


class ModelBackedEpisodePolicy(ModelBackedPolicyBase):
    """H1 (stateful) action policy backed by a real frozen model.

    Receives the rendered WORKINGMEMORY observation (ObservationRenderer)
    plus the raw EpisodeState. Same backend, same system prompt shape, same
    generation settings as ModelBackedFlatPolicy -- only the action set and
    observation content differ, per the harness's C1-C5 configuration.
    """

    def __init__(self, backend: TransformersModelBackend, max_parse_retries: int = 1):
        super().__init__(backend, H1_ALLOWED_ACTIONS, max_parse_retries)

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        prompt = (
            f"{self.system_prompt}\n"
            f"Valid actions: {self._action_list_text()}.\n"
            f"{observation}\n"
        )
        return self._generate_and_parse(prompt)
