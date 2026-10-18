import json
from pathlib import Path

import pytest

from src.dataset.schema import (
    CandidateElement,
    ExpectedAction,
    TraceStep,
    load_trace_dataset,
    validate_trace_step,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_trace_step_roundtrip_serialization():
    step = TraceStep(
        trace_id="550e8400-e29b-41d4-a716-446655440000",
        step_index=0,
        timestamp="2026-09-28T10:15:30Z",
        url="https://shop.corvus.dev/search",
        dom_pruned='[#1] body\n  [#2] input [type="search" placeholder="Search products"]\n  [#3] button "Search" [type="submit"]',
        instruction="Type 'wireless headphones' into the search box",
        expected_action=ExpectedAction(
            action_type="type",
            target_node_id=2,
            input_text="wireless headphones",
        ),
        candidates=[
            CandidateElement(node_id=2, role="searchbox", text=None),
            CandidateElement(node_id=3, role="button", text="Search"),
        ],
        metadata={
            "viewport_width": 1080,
            "viewport_height": 2400,
            "device_model": "Pixel 8 Pro",
        },
    )

    serialized = step.to_json()
    parsed_dict = json.loads(serialized)
    restored = TraceStep.from_dict(parsed_dict)

    assert restored.trace_id == step.trace_id
    assert restored.step_index == 0
    assert restored.url == "https://shop.corvus.dev/search"
    assert restored.expected_action.action_type == "type"
    assert restored.expected_action.target_node_id == 2
    assert restored.expected_action.input_text == "wireless headphones"
    assert len(restored.candidates) == 2
    assert validate_trace_step(restored) is True


def test_validate_trace_step_rejects_invalid_action_or_step_index():
    invalid_action_step = TraceStep(
        trace_id="550e8400-e29b-41d4-a716-446655440000",
        step_index=0,
        timestamp="2026-09-28T10:15:30Z",
        url="https://corvus.dev",
        dom_pruned="[#1] body",
        instruction="Invalid action test",
        expected_action=ExpectedAction(action_type="invalid_action", target_node_id=1),
        candidates=[CandidateElement(node_id=1, role="button", text="Go")],
    )
    with pytest.raises(ValueError, match="Invalid action_type"):
        validate_trace_step(invalid_action_step)

    negative_step = TraceStep(
        trace_id="550e8400-e29b-41d4-a716-446655440000",
        step_index=-1,
        timestamp="2026-09-28T10:15:30Z",
        url="https://corvus.dev",
        dom_pruned="[#1] body",
        instruction="Negative step index",
        expected_action=ExpectedAction(action_type="click", target_node_id=1),
        candidates=[CandidateElement(node_id=1, role="button", text="Go")],
    )
    with pytest.raises(ValueError, match="step_index"):
        validate_trace_step(negative_step)


@pytest.mark.parametrize(
    "relative_path",
    [
        "datasets/calibration/webvoyager_pruned_train.jsonl",
        "datasets/calibration/webvoyager_pruned_val.jsonl",
        "datasets/benchmarks/e-commerce_flows.jsonl",
        "datasets/benchmarks/auth_navigation.jsonl",
    ],
)
def test_reference_datasets_exist_and_conform_to_spec_data_001(relative_path: str):
    dataset_path = REPO_ROOT / relative_path
    assert dataset_path.exists(), (
        f"Missing dataset required by SPEC-DATA-001: {dataset_path}"
    )

    steps = load_trace_dataset(str(dataset_path))
    assert len(steps) >= 3, f"Expected at least 3 navigation steps in {relative_path}"

    for step in steps:
        assert validate_trace_step(step) is True
        assert "[#" in step.dom_pruned
        assert len(step.candidates) >= 1
        candidate_ids = {c.node_id for c in step.candidates}
        if step.expected_action.action_type != "noop":
            assert step.expected_action.target_node_id in candidate_ids
