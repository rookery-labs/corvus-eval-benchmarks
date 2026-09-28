from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json

@dataclass
class ExpectedAction:
    action_type: str
    target_node_id: int
    input_text: Optional[str] = None

@dataclass
class CandidateElement:
    node_id: int
    role: str
    text: Optional[str] = None

@dataclass
class TraceStep:
    trace_id: str
    step_index: int
    timestamp: str
    url: str
    dom_pruned: str
    instruction: str
    expected_action: ExpectedAction
    candidates: List[CandidateElement]
    metadata: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStep":
        expected = ExpectedAction(**data["expected_action"])
        candidates = [CandidateElement(**c) for c in data.get("candidates", [])]
        return cls(
            trace_id=data["trace_id"],
            step_index=data["step_index"],
            timestamp=data["timestamp"],
            url=data["url"],
            dom_pruned=data["dom_pruned"],
            instruction=data["instruction"],
            expected_action=expected,
            candidates=candidates,
            metadata=data.get("metadata"),
        )
