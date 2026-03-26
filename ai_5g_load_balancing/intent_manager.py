from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Intent:
    intent_id: int
    tenant: str
    slice_id: str
    latency_target_ms: float
    bandwidth_mbps: float
    priority: int = 1

    def to_dict(self) -> Dict:
        return {
            "intent_id": self.intent_id,
            "tenant": self.tenant,
            "slice_id": self.slice_id,
            "latency_target_ms": self.latency_target_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "priority": self.priority,
        }


class IntentManager:
    def __init__(self):
        self._intents: Dict[int, Intent] = {}
        self._id_gen = itertools.count(1)

    def add_intent(
        self,
        tenant: str,
        slice_id: str,
        latency_target_ms: float,
        bandwidth_mbps: float,
        priority: int = 1,
    ) -> Intent:
        intent = Intent(
            intent_id=next(self._id_gen),
            tenant=tenant,
            slice_id=slice_id,
            latency_target_ms=latency_target_ms,
            bandwidth_mbps=bandwidth_mbps,
            priority=priority,
        )
        self._intents[intent.intent_id] = intent
        return intent

    def remove_intent(self, intent_id: int) -> bool:
        return self._intents.pop(intent_id, None) is not None

    def list_intents(self) -> List[Intent]:
        return list(self._intents.values())

    def as_policy_dict(self) -> Dict[str, Dict]:
        policy = {}
        for intent in self._intents.values():
            policy[intent.slice_id] = {
                "latency_target_ms": intent.latency_target_ms,
                "bandwidth_mbps": intent.bandwidth_mbps,
                "priority": intent.priority,
            }
        return policy
