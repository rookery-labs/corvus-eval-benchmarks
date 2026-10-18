import os
import tempfile

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from src.quantization.calibrate_int8 import (
    SyntheticCalibrationDataReader,
    quantize_model_to_int8,
    validate_onnx_model,
)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_onnx_export_and_graph_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model_fp32.onnx")
        pytorch_model = SimpleClassifier().eval()
        dummy_input = torch.randn(1, 16)

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )

        assert os.path.exists(model_path)
        assert validate_onnx_model(model_path)

        # Validate onnxruntime execution
        session = ort.InferenceSession(model_path)
        test_in = np.random.randn(1, 16).astype(np.float32)
        out = session.run(None, {"input": test_in})
        assert len(out) == 1
        assert out[0].shape == (1, 4)


def test_static_int8_quantization_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, "classifier_fp32.onnx")
        int8_path = os.path.join(tmpdir, "classifier_int8.onnx")

        pytorch_model = SimpleClassifier().eval()
        dummy_input = torch.randn(1, 16)

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            fp32_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )

        reader = SyntheticCalibrationDataReader(
            input_name="input", input_shape=(1, 16), num_samples=10
        )
        quantize_model_to_int8(fp32_path, int8_path, reader)

        assert os.path.exists(int8_path)
        assert validate_onnx_model(int8_path)

        # Ensure quantized model runs in onnxruntime
        int8_session = ort.InferenceSession(int8_path)
        test_in = np.random.randn(1, 16).astype(np.float32)
        out = int8_session.run(None, {"input": test_in})
        assert len(out) == 1
        assert out[0].shape == (1, 4)
