from typing import Dict, Any, Union, List
import numpy as np
import torch


class BaseEngine:
    """Base class for inference engine"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        """Initialize inference engine.

        Args:
            model_path: Path to the model file
            config: Inference configuration
        """
        self.model_path = model_path
        self.config = config
        self.engine = None

    def load(self):
        """Load model"""
        raise NotImplementedError

    def preprocess(self, input_data: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Preprocess input data"""
        raise NotImplementedError

    def inference(self, input_data: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Run inference"""
        raise NotImplementedError

    def postprocess(self, output: List[np.ndarray]) -> Any:
        """Post-process results"""
        raise NotImplementedError

    def benchmark(self, input_shape: tuple, iterations: int = 100) -> Dict[str, float]:
        """Run performance benchmark"""
        raise NotImplementedError

    def release(self):
        """Release resources"""
        raise NotImplementedError
