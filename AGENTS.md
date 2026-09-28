# Corvus Eval Benchmarks — Agent Instructions & Domain Constraints

**Designated Primary Subagent:** `eval_bench_dev`  
**Quality Assurance Auditor:** `qa_sdd_validator`

---

## Technical Domain

This repository encapsulates the continuous evaluation, metrology, dataset curation, and static INT8 quantization pipelines for the Corvus browser ecosystem.

## Key Invariants & Rules

1. **Expected Calibration Error (ECE) & Metrology (`specs/ece-methodology.md`):**
   - Compute ECE and MCE across $M$ confidence bins.
   - Enforce deployment acceptance threshold: post-calibration $\text{ECE} \le 0.05$ (5.0%).
   - Optimize temperature parameter $T^*$ against validation sets.
2. **Static INT8 Quantization:**
   - Quantize float models into static INT8 QDQ format using ONNX Runtime Mobile.
   - Validate exported ONNX graphs with `onnx.checker` and run inference validation sessions.
3. **Trace Datasets (`specs/dataset-format.md`):**
   - Store navigation traces in JSONL format with structured `expected_action` and `candidates`.
4. **Testing Protocol:**
   - Run `pytest` within `.venv` before every commit.
