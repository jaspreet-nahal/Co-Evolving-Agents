from typing import Any, Dict, Optional

from .actions import ActionType, HarnessAction


class RuleBasedPolicy:

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        if state.turn == 0:
            return HarnessAction(ActionType.FAN_OUT_SEARCH, {"queries": [state.query]})
        if state.candidates and not state.curated:
            return HarnessAction.from_dict({"action": "curate", "arguments": {"add_ids": list(state.candidates)[:8], "importance": {item_id: "fair" for item_id in list(state.candidates)[:8]}}})
        unreviewed = [item_id for item_id in state.ordered_curated_ids() if item_id not in state.document_store]
        if unreviewed:
            return HarnessAction.from_dict({"action": "read_document", "arguments": {"doc_id": unreviewed[0]}})
        if state.turn >= 3 or state.curated:
            return HarnessAction.from_dict({"action": "end_search", "arguments": {"reason": "rule_based_policy_complete"}})
        return HarnessAction.from_dict({"action": "search_corpus", "arguments": {"query": state.query}})


class TransformersActionPolicy:

    def __init__(self, model_id: str = "google/gemma-4-31B-it", device_map: str = "auto", max_new_tokens: int = 512):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            from transformers import AutoModelForMultimodalLM
            self.model = AutoModelForMultimodalLM.from_pretrained(self.model_id, dtype="auto", device_map=self.device_map)
        except ImportError:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto", device_map=self.device_map)

    def choose_action(self, observation: str, state: Any) -> HarnessAction:
        self._load()
        prompt = (
            "You are a retrieval subagent. Return only JSON with keys action, arguments, reasoning.\n"
            "Valid actions: fan_out_search, search_corpus, grep_corpus, read_document, review_docs, curate, verify, end_search.\n"
            f"Query: {state.query}\n{observation}\n"
        )
        inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        try:
            return HarnessAction.from_json(text.strip())
        except (ValueError, TypeError):
            return RuleBasedPolicy().choose_action(observation, state)