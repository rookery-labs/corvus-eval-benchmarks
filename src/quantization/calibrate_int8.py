import os
from typing import List, Dict
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

class SyntheticCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, input_shape: tuple, num_samples: int = 20):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current = 0

    def get_next(self) -> Dict[str, np.ndarray]:
        if self.current >= self.num_samples:
            return None
        self.current += 1
        data = np.random.randn(*self.input_shape).astype(np.float32)
        return {self.input_name: data}

    def rewind(self):
        self.current = 0


def quantize_model_to_int8(
    input_model_path: str,
    output_model_path: str,
    data_reader: CalibrationDataReader,
) -> str:
    """
    Applies static INT8 quantization on an ONNX model.
    """
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    return output_model_path


def validate_onnx_model(model_path: str) -> bool:
    """
    Verifies ONNX model graph structure, node types, and inputs/outputs.
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return True
