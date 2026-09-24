from typing import Any, Dict, List

from .episode import EpisodeState


class ObservationRenderer:
    def __init__(self, max_chars: int = 30000, recent_actions: int = 5):
        self.max_chars = max_chars
        self.recent_actions = recent_actions

    def render(self, state: EpisodeState, latest_result: Dict[str, Any] = None) -> str:
        curated = []
        if state.curated_set is not None and state.candidate_pool is not None:
            for item_id in state.curated_set.ordered_curated_ids():
                candidate = state.candidate_pool.candidates.get(item_id)
                if candidate:
                    curated.append({
                        "id": item_id,
                        "importance": state.curated_set.curated[item_id].importance,
                        "snippet": candidate.snippet,
                    })

        pool = []
        if state.candidate_pool is not None:
            curated_ids = set(state.curated_set.curated) if state.curated_set is not None else set()
            for item_id, candidate in list(state.candidate_pool.candidates.items())[-50:]:
                if item_id not in curated_ids:
                    pool.append({"id": item_id, "score": round(candidate.score, 4), "snippet": candidate.snippet})

        graph = []
        if state.evidence_graph is not None:
            graph = [
                {"entity": entity, "documents": sorted(doc_ids)}
                for entity, doc_ids in state.evidence_graph.edges.items()
                if len(doc_ids) > 1
            ][:8]

        verification = []
        if state.verification_cache is not None:
            verification = list(state.verification_cache.records)[-10:]

        sufficiency = None
        if state.sufficiency is not None and state.sufficiency.history:
            latest = state.sufficiency.history[-1]
            sufficiency = {"decision": latest.decision, "reason": latest.reason, "confidence": latest.confidence}

        payload = {
            "query": state.query,
            "turn": state.turn,
            "budget": {"turns_used": state.turn, "turns_remaining": max(state.max_turns - state.turn, 0)},
            "enabled_components": {
                "candidate_pool": state.enable_candidate_pool,
                "curated_set": state.enable_curated_set,
                "evidence_graph": state.enable_evidence_graph,
                "verification_cache": state.enable_verification_cache,
                "sufficiency_check": state.enable_sufficiency_check,
            },
            "curated_set": curated,
            "candidate_pool": pool,
            "evidence_graph": graph,
            "verification": verification,
            "sufficiency": sufficiency,
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
