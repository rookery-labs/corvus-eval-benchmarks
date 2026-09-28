# Corvus Eval Benchmarks

Evaluation benches, navigation trace datasets, Expected Calibration Error (ECE) metrology, and static INT8 quantization pipelines for the **Corvus** browser ecosystem.

## Components

- **Calibration & Metrics (`src/metrics/ece.py`):** Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and scalar temperature optimization.
- **Trace Datasets (`src/dataset/schema.py`):** Structured JSONL dataset schemas for offline/online evaluation of agent navigation traces.
- **Quantization Pipeline (`src/quantization/calibrate_int8.py`):** INT8 post-training static calibration with ONNX Runtime Mobile QDQ format and graph verification.

## Specifications (SDD)

- [ECE Methodology & Metrology](specs/ece-methodology.md)
- [Navigation Trace Dataset Schema](specs/dataset-format.md)

## Development & Testing (TDD)

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -e .

# Run pytest test suite
pytest
```

## License

Licensed under the [Apache License 2.0](LICENSE).
