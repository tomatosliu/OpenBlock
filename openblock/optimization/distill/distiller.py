import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner


@MODELS.register_module()
class KnowledgeDistiller(BaseModel):
    """Knowledge Distillation Model based on MMEngine.

    This model implements knowledge distillation training using MMEngine's training pipeline.
    It supports various distillation methods including vanilla KD, feature-based KD, etc.

    Args:
        teacher (dict): Config for teacher model
        student (dict): Config for student model
        distill_cfg (dict): Config for distillation, including:
            - temperature: Temperature for softmax
            - alpha: Weight for distillation loss
            - kd_loss: Type of KD loss
        data_preprocessor (Optional[dict]): Config for data preprocessor
        init_cfg (Optional[dict]): Config for initialization
    """

    def __init__(self,
                 teacher: dict,
                 student: dict,
                 distill_cfg: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.teacher = MODELS.build(teacher)
        self.student = MODELS.build(student)
        self.distill_cfg = distill_cfg

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[Any]] = None,
                mode: str = 'tensor') -> Dict[str, torch.Tensor]:
        """Forward computation.

        Args:
            inputs (torch.Tensor): Input images
            data_samples (Optional[List[Any]]): Data samples
            mode (str): Forward mode, 'tensor', 'loss' or 'predict'

        Returns:
            Dict[str, torch.Tensor]: Forward results
        """
        if mode == 'loss':
            return self.compute_loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.student(inputs)

    def compute_loss(self,
                     inputs: torch.Tensor,
                     data_samples: List[Any]) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            inputs (torch.Tensor): Input images
            data_samples (List[Any]): Data samples

        Returns:
            Dict[str, torch.Tensor]: Loss dict
        """
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        # Get student outputs
        student_outputs = self.student(inputs)

        # Compute distillation loss
        losses = dict()

        # KL divergence loss for logits
        T = self.distill_cfg.get('temperature', 1.0)
        alpha = self.distill_cfg.get('alpha', 0.5)

        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / T, dim=1),
            F.softmax(teacher_outputs / T, dim=1),
            reduction='batchmean') * (T * T)

        # Hard loss with ground truth
        hard_loss = F.cross_entropy(student_outputs, data_samples)

        # Total loss
        losses['loss'] = alpha * soft_loss + (1 - alpha) * hard_loss
        losses['soft_loss'] = soft_loss
        losses['hard_loss'] = hard_loss

        return losses

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[Any]] = None) -> List[Any]:
        """Inference without augmentation.

        Args:
            inputs (torch.Tensor): Input images
            data_samples (Optional[List[Any]]): Data samples

        Returns:
            List[Any]: Prediction results
        """
        return self.student.predict(inputs, data_samples)

    def train_step(self, data: Dict[str, Any], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            data (Dict[str, Any]): Data from dataloader
            optim_wrapper (OptimWrapper): Optimizer wrapper

        Returns:
            Dict[str, torch.Tensor]: Loss dict
        """
        # Process data
        data = self.data_preprocessor(data, True)

        # Forward
        losses = self(**data, mode='loss')

        # Optimize
        optim_wrapper.update_params(losses)

        return losses

    @staticmethod
    def build_runner(cfg: dict) -> Runner:
        """Build MMEngine runner.

        Args:
            cfg (dict): Config dict

        Returns:
            Runner: MMEngine runner
        """
        return Runner.from_cfg(cfg)
