import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExpectedAction:
    action_type: str
    target_node_id: int
    input_text: str | None = None


@dataclass
class CandidateElement:
    node_id: int
    role: str
    text: str | None = None


@dataclass
class TraceStep:
    trace_id: str
    step_index: int
    timestamp: str
    url: str
    dom_pruned: str
    instruction: str
    expected_action: ExpectedAction
    candidates: list[CandidateElement]
    metadata: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
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


VALID_ACTION_TYPES = {"click", "type", "clear", "scroll", "select", "noop"}


def validate_trace_step(step: TraceStep) -> bool:
    if not step.trace_id or not isinstance(step.trace_id, str):
        raise ValueError("trace_id must be a non-empty string")
    if not isinstance(step.step_index, int) or step.step_index < 0:
        raise ValueError(
            f"step_index must be a non-negative integer, got {step.step_index}"
        )
    if not step.timestamp or not isinstance(step.timestamp, str):
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    if not step.url or not (
        step.url.startswith("http://") or step.url.startswith("https://")
    ):
        raise ValueError(f"url must be a valid HTTP/HTTPS URI, got {step.url}")
    if not step.dom_pruned or not isinstance(step.dom_pruned, str):
        raise ValueError("dom_pruned must be a non-empty string")
    if not step.instruction or not isinstance(step.instruction, str):
        raise ValueError("instruction must be a non-empty string")
    if step.expected_action.action_type not in VALID_ACTION_TYPES:
        raise ValueError(
            f"Invalid action_type '{step.expected_action.action_type}'; must be one of {sorted(VALID_ACTION_TYPES)}"
        )
    if not isinstance(step.expected_action.target_node_id, int):
        raise TypeError("expected_action.target_node_id must be an integer")
    if not isinstance(step.candidates, list) or len(step.candidates) == 0:
        raise ValueError("candidates must be a non-empty list")
    for cand in step.candidates:
        if not isinstance(cand.node_id, int) or not cand.role:
            raise ValueError(
                "Each candidate must have an integer node_id and non-empty role"
            )
    return True


def load_trace_dataset(jsonl_path: str) -> list[TraceStep]:
    steps: list[TraceStep] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            step = TraceStep.from_dict(record)
            validate_trace_step(step)
            steps.append(step)
    return steps
