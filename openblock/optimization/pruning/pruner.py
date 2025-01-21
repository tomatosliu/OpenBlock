import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class Pruner:
    """Base class for model pruning"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize pruner.

        Args:
            model: Model to be pruned
            config: Pruning configuration, including pruning ratio and other parameters
        """
        self.model = model
        self.config = config
        self.pruned_model = None

    def analyze(self) -> Dict[str, Any]:
        """Analyze model structure and parameter distribution"""
        raise NotImplementedError

    def prune(self) -> nn.Module:
        """Execute model pruning"""
        raise NotImplementedError

    def fine_tune(self,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: Optional[torch.utils.data.DataLoader] = None,
                  epochs: int = 10) -> nn.Module:
        """Fine-tune pruned model"""
        raise NotImplementedError

    def export(self, path: str, format: str = 'onnx'):
        """Export pruned model"""
        raise NotImplementedError
