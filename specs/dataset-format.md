# Specification: Navigation Trace Dataset Schema & Storage Format

**Spec ID:** `SPEC-DATA-001`  
**Status:** VALIDATED  
**Component:** `corvus-eval-benchmarks`

---

## 1. Storage Architecture

Navigation traces are recorded as newline-delimited JSON (`JSONL`) files (`*.jsonl`) organized by evaluation suites:
```
datasets/
├── calibration/
│   ├── webvoyager_pruned_train.jsonl
│   └── webvoyager_pruned_val.jsonl
└── benchmarks/
    ├── e-commerce_flows.jsonl
    └── auth_navigation.jsonl
```

---

## 2. Record JSON Schema

Each line in a dataset file corresponds to a single decision point during browser navigation:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CorvusNavigationTraceStep",
  "type": "object",
  "required": [
    "trace_id",
    "step_index",
    "timestamp",
    "url",
    "dom_pruned",
    "instruction",
    "expected_action",
    "candidates"
  ],
  "properties": {
    "trace_id": { "type": "string", "format": "uuid" },
    "step_index": { "type": "integer", "minimum": 0 },
    "timestamp": { "type": "string", "format": "date-time" },
    "url": { "type": "string", "format": "uri" },
    "dom_pruned": {
      "type": "string",
      "description": "Standardized indented representation emitted by corvus-dom-extension"
    },
    "instruction": {
      "type": "string",
      "description": "Natural language user goal or immediate sub-goal"
    },
    "expected_action": {
      "type": "object",
      "required": ["action_type", "target_node_id"],
      "properties": {
        "action_type": {
          "type": "string",
          "enum": ["click", "type", "clear", "scroll", "select", "noop"]
        },
        "target_node_id": { "type": "integer" },
        "input_text": { "type": "string" }
      }
    },
    "candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["node_id", "role"],
        "properties": {
          "node_id": { "type": "integer" },
          "role": { "type": "string" },
          "text": { "type": "string" }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "viewport_width": { "type": "integer" },
        "viewport_height": { "type": "integer" },
        "device_model": { "type": "string" }
      }
    }
  }
}
```
