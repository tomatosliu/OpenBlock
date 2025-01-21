import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np

from openblock.optimization.pruning import Pruner
from openblock.optimization.distill import Distiller
from openblock.optimization.quant import Quantizer
from openblock.inference.tensorrt import TRTEngine


def main():
    # 1. Prepare models
    teacher_model = resnet18(pretrained=True)
    student_model = resnet18(num_classes=1000)  # Simplified version

    # 2. Knowledge distillation
    distill_config = {
        'temperature': 4.0,
        'alpha': 0.5,  # Weight for distillation loss
        'epochs': 100,
        'learning_rate': 1e-4
    }

    distiller = Distiller(
        teacher_model=teacher_model,
        student_model=student_model,
        config=distill_config
    )

    # Assume we have training data loaders
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)

    optimized_model = distiller.distill(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # 3. Model pruning
    prune_config = {
        'method': 'l1',
        'ratio': 0.5,  # Pruning ratio
        'fine_tune_epochs': 10
    }

    pruner = Pruner(
        model=optimized_model,
        config=prune_config
    )

    pruned_model = pruner.prune()
    pruned_model = pruner.fine_tune(train_loader, val_loader)

    # 4. Model quantization
    quant_config = {
        'backend': 'tensorrt',
        'precision': 'int8',
        'calibration_size': 100
    }

    quantizer = Quantizer(
        model=pruned_model,
        config=quant_config
    )

    # Use calibration dataset for quantization
    quantizer.calibrate(val_loader)
    quantized_model = quantizer.quantize()

    # Export model
    quantizer.export('optimized_model.trt', format='tensorrt')

    # 5. TensorRT inference
    engine_config = {
        'max_batch_size': 32,
        'workspace_size': 1 << 30  # 1GB
    }

    engine = TRTEngine(
        model_path='optimized_model.trt',
        config=engine_config
    )

    engine.load()

    # Run inference
    dummy_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
    output = engine.inference(dummy_input)

    # Performance testing
    benchmark_results = engine.benchmark(
        input_shape=(1, 3, 224, 224),
        iterations=100
    )

    print("Benchmark results:", benchmark_results)

    # Release resources
    engine.release()


if __name__ == '__main__':
    main()
