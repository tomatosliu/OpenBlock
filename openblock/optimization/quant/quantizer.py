import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List


class Quantizer:
    """Base class for model quantization"""

    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any]):
        """Initialize quantizer.

        Args:
            model: Model to be quantized
            config: Quantization configuration, including bit width, quantization scheme, etc.
        """
        self.model = model
        self.config = config
        self.quantized_model = None

    def calibrate(self,
                  calibration_loader: torch.utils.data.DataLoader):
        """Calibrate quantization parameters using calibration dataset"""
        raise NotImplementedError

    def quantize(self) -> nn.Module:
        """Execute model quantization"""
        raise NotImplementedError

    def evaluate(self,
                 val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate quantized model performance"""
        raise NotImplementedError

    def export(self,
               path: str,
               format: str = 'tensorrt',
               input_shapes: Optional[List[tuple]] = None):
        """Export quantized model"""
        raise NotImplementedError
