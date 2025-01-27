import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmengine.structures import BaseDataElement


class DistillationLoss:
    """Knowledge distillation loss functions.

    Supports various distillation losses including:
    - KL divergence
    - Mean squared error
    - Feature-level distillation
    """

    @staticmethod
    def kl_div(student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
        """KL divergence loss for knowledge distillation.

        Args:
            student_logits (torch.Tensor): Student model logits
            teacher_logits (torch.Tensor): Teacher model logits
            temperature (float): Temperature for softening probability distributions

        Returns:
            torch.Tensor: KL divergence loss
        """
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        log_probs = F.log_softmax(student_logits / temperature, dim=1)
        loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        return loss * (temperature ** 2)

    @staticmethod
    def mse(student_features: torch.Tensor,
            teacher_features: torch.Tensor) -> torch.Tensor:
        """Mean squared error loss for feature-level distillation.

        Args:
            student_features (torch.Tensor): Student model features
            teacher_features (torch.Tensor): Teacher model features

        Returns:
            torch.Tensor: MSE loss
        """
        return F.mse_loss(student_features, teacher_features)

    @staticmethod
    def attention(student_features: torch.Tensor,
                  teacher_features: torch.Tensor) -> torch.Tensor:
        """Attention transfer loss for feature-level distillation.

        Args:
            student_features (torch.Tensor): Student model features
            teacher_features (torch.Tensor): Teacher model features

        Returns:
            torch.Tensor: Attention transfer loss
        """
        student_attention = F.normalize(student_features.pow(
            2).mean(1).view(student_features.size(0), -1))
        teacher_attention = F.normalize(teacher_features.pow(
            2).mean(1).view(teacher_features.size(0), -1))
        return (student_attention - teacher_attention).pow(2).mean()


@MODELS.register_module()
class DistillationModel(BaseModel):
    """Model for knowledge distillation training.

    This model wraps teacher and student models and implements
    the knowledge distillation training logic.

    Args:
        teacher_model (nn.Module): Pre-trained teacher model
        student_model (nn.Module): Student model to be trained
        distill_cfg (Dict): Distillation configuration
        train_cfg (Optional[Dict]): Training configuration
        test_cfg (Optional[Dict]): Testing configuration
    """

    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 distill_cfg: Dict,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None):
        super().__init__()

        self.teacher = teacher_model
        self.student = student_model
        self.distill_cfg = distill_cfg

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Loss weights
        self.alpha = distill_cfg.get('alpha', 0.5)
        self.temperature = distill_cfg.get('temperature', 1.0)

        # Feature distillation settings
        self.feature_distill = distill_cfg.get('feature_distill', False)
        if self.feature_distill:
            self.feature_layers = distill_cfg['feature_layers']
            self.feature_loss_weight = distill_cfg.get(
                'feature_loss_weight', 1.0)

            # Create adaptation layers for each feature layer
            self.adaptation_layers = nn.ModuleDict()

            # Get feature dimensions
            with torch.no_grad():
                # Use a dummy input to get feature dimensions
                dummy_input = torch.randn(1, 3, 224, 224)
                teacher_features = self.extract_features(
                    self.teacher, dummy_input, self.feature_layers)
                student_features = self.extract_features(
                    self.student, dummy_input, self.feature_layers)

                for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
                    layer_name = f'adapt_layer{i+1}'
                    # Create 1x1 conv to match channel dimensions
                    self.adaptation_layers[layer_name] = nn.Conv2d(
                        s_feat.shape[1], t_feat.shape[1],
                        kernel_size=1, stride=1, padding=0, bias=False
                    )

    def extract_features(self, model: nn.Module, x: torch.Tensor, layer_names: List[str]) -> List[torch.Tensor]:
        """Extract intermediate features from specified layers.

        Args:
            model (nn.Module): Model to extract features from
            x (torch.Tensor): Input tensor
            layer_names (List[str]): Names of layers to extract features from

        Returns:
            List[torch.Tensor]: List of feature tensors
        """
        features = []

        # Get the backbone
        backbone = model.backbone if hasattr(model, 'backbone') else model

        # Initial layers
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        # Layer blocks
        layer_mapping = {
            'backbone.layer1': backbone.layer1,
            'backbone.layer2': backbone.layer2,
            'backbone.layer3': backbone.layer3,
            'backbone.layer4': backbone.layer4
        }

        for name, layer in layer_mapping.items():
            if name in layer_names:
                x = layer(x)
                features.append(x)
            else:
                x = layer(x)

        return features

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward computation.

        Args:
            inputs (torch.Tensor): Input tensor
            data_samples (list): List of data samples
            mode (str): Forward mode, 'tensor', 'loss', or 'predict'

        Returns:
            Union[torch.Tensor, dict]: Forward results
        """
        if mode == 'loss':
            return self.forward_train(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_tensor(inputs)
        else:
            raise ValueError(f'Invalid mode: {mode}')

    def forward_train(self, inputs, data_samples):
        """Forward computation during training.

        Args:
            inputs (torch.Tensor): Input tensor
            data_samples (list): List of data samples

        Returns:
            dict: Loss dict
        """
        # Get teacher outputs
        with torch.no_grad():
            teacher_logits = self.teacher(inputs, mode='tensor')
            if self.feature_distill:
                teacher_features = self.extract_features(
                    self.teacher, inputs, self.feature_layers)

        # Get student outputs
        student_logits = self.student(inputs, mode='tensor')
        student_features = self.extract_features(
            self.student, inputs, self.feature_layers) if self.feature_distill else []

        # Get ground truth labels
        gt_labels = torch.tensor([
            sample.gt_label for sample in data_samples
        ], device=inputs.device)

        # Compute losses
        losses = {}

        # Hard label loss
        hard_loss = nn.functional.cross_entropy(student_logits, gt_labels)
        losses['hard_loss'] = hard_loss

        # Soft label loss (KL divergence)
        soft_loss = DistillationLoss.kl_div(
            student_logits, teacher_logits, self.temperature)
        losses['soft_loss'] = soft_loss

        # Feature distillation loss if enabled
        if self.feature_distill and student_features:
            feat_loss = 0
            for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
                # Apply adaptation layer
                layer_name = f'adapt_layer{i+1}'
                adapted_s_feat = self.adaptation_layers[layer_name](s_feat)
                # Compute MSE loss
                feat_loss += nn.functional.mse_loss(adapted_s_feat, t_feat)
            feat_loss /= len(student_features)
            losses['feat_loss'] = feat_loss * self.feature_loss_weight

        # Total loss
        losses['loss'] = (1 - self.alpha) * hard_loss + \
            self.alpha * soft_loss + \
            losses.get('feat_loss', 0.0)

        return losses

    def forward_predict(self, inputs, data_samples=None):
        """Forward computation during inference.

        Args:
            inputs (torch.Tensor): Input tensor
            data_samples (list, optional): List of data samples

        Returns:
            list: List of predictions
        """
        student_logits = self.student(inputs, mode='tensor')
        predictions = torch.argmax(student_logits, dim=1)

        # Update data samples with predictions
        for sample, pred in zip(data_samples, predictions):
            sample.pred_label = pred.item()

        return data_samples

    def forward_tensor(self, inputs):
        """Forward computation to get raw logits.

        Args:
            inputs (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output logits
        """
        return self.student(inputs, mode='tensor')

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

        # Forward and get losses
        losses = self(**data, mode='loss')

        # Parse losses
        loss = losses['loss']  # Get total loss for backward
        parsed_losses = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in losses.items()
        }

        # Optimize
        optim_wrapper.update_params(loss)

        return parsed_losses

    @staticmethod
    def build_runner(cfg: dict) -> Runner:
        """Build MMEngine runner.

        Args:
            cfg (dict): Config dict

        Returns:
            Runner: MMEngine runner
        """
        return Runner.from_cfg(cfg)
