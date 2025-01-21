import argparse
import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from openblock.inference.tensorrt import TRTEngine


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy model to TensorRT')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--output', help='Output TensorRT engine path', default='model.trt')
    parser.add_argument('--device', help='Device to use', default='cuda:0')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 mode')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Load model
    model = cfg.model.student
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(args.device)
    model.eval()

    # Export to ONNX first
    dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    torch.onnx.export(model, dummy_input, 'temp.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Convert to TensorRT
    engine_config = {
        'max_batch_size': 32,
        'workspace_size': 1 << 30,  # 1GB
        'fp16_mode': args.fp16
    }

    engine = TRTEngine('temp.onnx', engine_config)
    engine.build(args.output)

    # Test inference
    print("Testing inference...")
    input_data = np.random.random((1, 3, 32, 32)).astype(np.float32)
    output = engine.inference(input_data)

    # Benchmark
    print("Running benchmark...")
    benchmark_results = engine.benchmark(
        input_shape=(1, 3, 32, 32),
        iterations=100
    )
    print("Benchmark results:", benchmark_results)

    engine.release()


if __name__ == '__main__':
    main()
