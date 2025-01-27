# OpenBlock - Efficient Vision Model Training and Optimization Platform

A comprehensive platform for training and optimizing vision models with a focus on efficiency and minimal data requirements. Built on MMlab ecosystem, this platform specializes in knowledge distillation, efficient data usage, and model optimization.

## Key Features

### 1. Efficient Training Framework
- **Knowledge Distillation**
  - Feature-level and logit-level distillation
  - Flexible teacher-student architecture
  - Multiple distillation loss functions
  
- **Data Efficiency**
  - Smart sampling strategies for minimal data usage
  - Balanced class sampling for imbalanced datasets
  - Diversity-based sampling for active learning
  
- **Data Augmentation**
  - Task-specific augmentation pipelines
  - Teacher-student specific transforms
  - Configurable augmentation strategies

### 2. Model Optimization
- **Knowledge Transfer**
  - Teacher-student model distillation
  - Feature map attention transfer
  - Intermediate layer supervision
  
- **Training Efficiency**
  - Efficient data loading and caching
  - Flexible model architectures
  - Customizable training pipelines

### 3. Visualization and Monitoring
- **Training Progress**
  - Real-time loss curve visualization
  - Learning rate scheduling plots
  - Model performance comparison
  
- **Model Analysis**
  - Feature map visualization
  - Confusion matrix analysis
  - Model size and FLOPs comparison

## Project Structure
```
OpenBlock/
├── openblock/
│   ├── datasets/           # Dataset implementations
│   │   ├── base_dataset.py     # Base dataset class
│   │   ├── converters.py       # Format conversion utilities
│   │   ├── pipelines.py        # Data augmentation pipelines
│   │   └── samplers.py         # Sampling strategies
│   ├── optimization/      # Optimization algorithms
│   │   └── distill/          # Knowledge distillation
│   └── visualization/     # Visualization tools
│       └── training_monitor.py  # Training progress visualization
├── examples/             # Usage examples
│   └── classification_distillation.py  # Image classification example
└── requirements.txt      # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/OpenBlock.git
cd OpenBlock
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Image Classification with Knowledge Distillation

```python
from openblock.datasets import CIFAR10Dataset
from openblock.datasets.pipelines import ImageClassificationTransform
from openblock.optimization.distill import DistillationModel

# Create datasets with augmentation
train_transform = ImageClassificationTransform(
    img_size=224,
    is_training=True
)

train_dataset = CIFAR10Dataset(
    data_root='data/cifar10',
    data_prefix=dict(img='train'),
    pipeline=train_transform
)

# Create teacher and student models
teacher_model = SimpleClassifier(backbone='resnet50', num_classes=10)
student_model = SimpleClassifier(backbone='resnet18', num_classes=10)

# Setup distillation
model = DistillationModel(
    teacher_model=teacher_model,
    student_model=student_model,
    distill_cfg=dict(
        alpha=0.5,
        temperature=4.0,
        feature_distill=True
    )
)

# Train with visualization
runner = Runner(
    model=model,
    train_dataloader=train_loader,
    # ... other configurations
)
runner.train()
```

### 2. Custom Dataset Support

```python
from openblock.datasets import OpenBlockDataset
from openblock.datasets.converters import DatasetConverter

# Convert from COCO format
annotations = DatasetConverter.coco_to_openblock(
    coco_file='annotations.json',
    img_dir='images',
    task_type='detection'
)

# Convert from VOC format
annotations = DatasetConverter.voc_to_openblock(
    voc_dir='annotations',
    img_dir='images'
)
```

### 3. Training Visualization

```python
from openblock.visualization import TrainingMonitor

monitor = TrainingMonitor(
    name='experiment_name',
    use_wandb=True  # Enable Weights & Biases logging
)

# Plot training progress
monitor.plot_losses(losses, steps)
monitor.plot_lr_schedule(learning_rates, steps)
monitor.plot_confusion_matrix(conf_matrix, class_names)
```

## Examples

Check the `examples/` directory for complete usage examples:
- `classification_distillation.py`: Image classification with knowledge distillation
- More examples coming soon...

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

<!-- ## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use OpenBlock in your research, please cite:
```bibtex
@misc{openblock2024,
  author = {Your Name},
  title = {OpenBlock: Efficient Vision Model Training and Optimization Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/OpenBlock}
} -->
```