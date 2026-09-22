from typing import Any, Dict, List

from .episode import EpisodeState


class ObservationRenderer:
    def __init__(self, max_chars: int = 30000, recent_actions: int = 5):
        self.max_chars = max_chars
        self.recent_actions = recent_actions

    def render(self, state: EpisodeState, latest_result: Dict[str, Any] = None) -> str:
        curated = []
        for item_id in state.ordered_curated_ids():
            candidate = state.candidates.get(item_id)
            if candidate:
                curated.append({"id": item_id, "importance": state.curated[item_id].importance, "snippet": candidate.snippet})
        pool = []
        for item_id, candidate in list(state.candidates.items())[-50:]:
            if item_id not in state.curated:
                pool.append({"id": item_id, "score": round(candidate.score, 4), "snippet": candidate.snippet})
        graph = [
            {"entity": entity, "documents": sorted(doc_ids)}
            for entity, doc_ids in state.evidence_graph.items()
            if len(doc_ids) > 1
        ][:8]
        payload = {
            "query": state.query,
            "turn": state.turn,
            "budget": {"turns_used": state.turn, "turns_remaining": max(state.max_turns - state.turn, 0)},
            "curated_set": curated,
            "candidate_pool": pool,
            "evidence_graph": graph,
            "verification": list(state.verification_cache)[-10:],
            "recent_actions": [
                {"turn": event.turn, "action": event.action, "result": event.result_summary}
                for event in state.action_history[-self.recent_actions:]
            ],
            "latest_result": latest_result or {},
        }
        text = "WORKINGMEMORY\n" + self._format(payload)
        return text[: self.max_chars]

    def _format(self, value: Any, indent: int = 0) -> str:
        if isinstance(value, dict):
            return "\n".join(" " * indent + f"{key}:\n" + self._format(item, indent + 2) for key, item in value.items())
        if isinstance(value, list):
            return "\n".join(" " * indent + f"- {self._format(item, 0)}" for item in value) or " " * indent + "- none"
        return " " * indent + str(value)