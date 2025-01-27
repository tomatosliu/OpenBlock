import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

from openblock.datasets import CIFAR10Dataset
from openblock.datasets.pipelines import (
    LoadImageFromFile, ImageClassificationTransform, PackClassificationInputs
)
from openblock.optimization.distill import DistillationModel
from openblock.datasets.samplers import BalancedClassSampler
from openblock.visualization import TrainingMonitor
from openblock.registry import MODELS
from openblock.evaluation.metrics import Accuracy

from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.config import Config
from mmengine.structures import BaseDataElement
from mmengine.evaluator import BaseMetric
from mmengine.evaluator import Evaluator


@MODELS.register_module()
class SimpleClassifier(BaseModel):
    """Simple classifier wrapper for torchvision models."""

    def __init__(self, backbone: str, num_classes: int):
        super().__init__()

        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor, data_samples=None, mode='tensor'):
        """Forward computation.

        Args:
            inputs (torch.Tensor): Input tensor
            data_samples: Additional data samples
            mode (str): Forward mode, 'tensor', 'loss', or 'predict'

        Returns:
            Union[torch.Tensor, dict]: Forward results
        """
        if mode == 'tensor':
            x = inputs
            return self.backbone(x)
        elif mode == 'predict':
            x = inputs
            logits = self.backbone(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions
        elif mode == 'loss':
            x = inputs
            logits = self.backbone(x)
            losses = dict()

            # Get ground truth labels
            gt_labels = torch.tensor([
                sample.gt_label for sample in data_samples
            ], device=logits.device)

            # Compute cross entropy loss
            loss = nn.functional.cross_entropy(logits, gt_labels)
            losses['loss'] = loss

            return losses

    def extract_feat(self, x):
        """Extract features before the final FC layer."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def custom_collate_fn(batch):
    """Custom collate function to handle LabelData objects.

    Args:
        batch (list): List of samples from dataset

    Returns:
        dict: Collated batch with inputs and data_samples
    """
    collated = {}

    # Collate inputs (images)
    collated['inputs'] = torch.stack([sample['inputs'] for sample in batch])

    # Collate data_samples
    collated['data_samples'] = [sample['data_samples'][0] for sample in batch]

    return collated


def main():
    # Configuration
    cfg = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': {
            'batch_size': 128,
            'num_workers': 4,
            'img_size': 224
        },
        'model': {
            'teacher_backbone': 'resnet50',
            'student_backbone': 'resnet18',
            'num_classes': 10
        },
        'distillation': {
            'alpha': 0.5,
            'temperature': 4.0,
            'feature_distill': True,
            'feature_layers': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3'],
            'feature_loss_weight': 0.1
        },
        'optimizer': {
            'type': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': 200
        },
        'total_epochs': 200
    }

    # Set random seed
    torch.manual_seed(cfg['seed'])

    # Create transform pipelines
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='ImageClassificationTransform',
            size=cfg['data']['img_size'],
            is_train=True
        ),
        dict(type='PackClassificationInputs')
    ]

    val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='ImageClassificationTransform',
            size=cfg['data']['img_size'],
            is_train=False
        ),
        dict(type='PackClassificationInputs')
    ]

    # Create datasets
    train_dataset = CIFAR10Dataset(
        data_root='data/cifar10',
        data_prefix=dict(img='train'),
        pipeline=train_pipeline,
        test_mode=False
    )

    val_dataset = CIFAR10Dataset(
        data_root='data/cifar10',
        data_prefix=dict(img='test'),
        pipeline=val_pipeline,
        test_mode=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=custom_collate_fn
    )

    # Create models
    teacher_model = SimpleClassifier(
        backbone=cfg['model']['teacher_backbone'],
        num_classes=cfg['model']['num_classes']
    )

    student_model = SimpleClassifier(
        backbone=cfg['model']['student_backbone'],
        num_classes=cfg['model']['num_classes']
    )

    # Create distillation model
    model = DistillationModel(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_cfg=cfg['distillation']
    )

    # Create runner
    runner = Runner(
        model=model,
        work_dir='work_dirs/cifar10_distillation',
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_cfg=dict(by_epoch=True, max_epochs=cfg['total_epochs']),
        val_cfg=dict(),
        val_evaluator=dict(
            type='Accuracy',
            topk=(1, 5)
        ),
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=cfg['optimizer']['lr'],
                momentum=cfg['optimizer']['momentum'],
                weight_decay=cfg['optimizer']['weight_decay']
            )
        ),
        param_scheduler=dict(
            type='CosineAnnealingLR',
            T_max=cfg['scheduler']['T_max'],
            by_epoch=True
        ),
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=100),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
            sampler_seed=dict(type='DistSamplerSeedHook'),
        ),
        custom_hooks=[],
        launcher='none',
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl'),
        )
    )

    # Train model
    runner.train()

    # Evaluate final model
    runner.val()

    # Save optimized student model
    torch.save(student_model.state_dict(),
               'work_dirs/cifar10_distillation/student_final.pth')


if __name__ == '__main__':
    main()
