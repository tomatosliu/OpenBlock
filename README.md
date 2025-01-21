# OpenBlock - Deep Learning Model Optimization and Inference Toolkit

A vision-based deep learning model optimization toolkit built on MMlab, focusing on model quantization, pruning, knowledge distillation, and efficient deployment.

## Key Features

- **Model Optimization**
  - Model Quantization: Support for INT8/FP16 quantization
  - Model Pruning: Structured/Unstructured pruning
  - Knowledge Distillation: Teacher-student model distillation framework
  
- **Inference Framework**
  - TensorRT inference support
  - RKNN inference support
  - Unified Python/C++ interface
  
- **Visualization Tools**
  - Training process visualization
  - Model performance comparison
  - Optimization effect visualization

## Project Structure

```
OpenBlock/
├── openblock/
│   ├── core/           # Core functionality implementation
│   ├── models/         # Model definitions
│   ├── datasets/       # Dataset processing
│   ├── optimization/   # Optimization algorithms
│   │   ├── pruning/    # Model pruning
│   │   ├── distill/    # Knowledge distillation
│   │   └── quant/      # Model quantization
│   ├── inference/      # Inference engines
│   │   ├── tensorrt/   # TensorRT inference
│   │   └── rknn/       # RKNN inference
│   └── visualization/  # Visualization tools
├── examples/           # Usage examples
├── tests/             # Unit tests
└── tools/             # Utility scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Model Optimization**
```python
from openblock.optimization import Pruner, Distiller, Quantizer

# Model pruning
pruner = Pruner(model, config)
pruned_model = pruner.prune()

# Knowledge distillation
distiller = Distiller(teacher_model, student_model, config)
optimized_model = distiller.distill()

# Model quantization
quantizer = Quantizer(model, config)
quantized_model = quantizer.quantize()
```

2. **Model Inference**
```python
from openblock.inference import TRTEngine, RKNNEngine

# TensorRT inference
engine = TRTEngine(model_path)
result = engine.inference(input_data)

# RKNN inference
engine = RKNNEngine(model_path)
result = engine.inference(input_data)
```

## Documentation

For detailed documentation, please refer to [docs/](docs/)

## Contributing

Issues and Pull Requests are welcome!

## License

This project is licensed under the Apache 2.0 License