import argparse
import torch
import numpy as np
import onnxruntime as ort
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deploy model using ONNX Runtime')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--output', help='Output ONNX model path', default='model.onnx')
    parser.add_argument('--device', help='Device to use', default='mps')
    return parser.parse_args()


class ONNXRuntimeEngine:
    def __init__(self, model_path: str):
        # Create ONNX Runtime session
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def inference(self, input_data: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: input_data})[0]

    def benchmark(self, input_shape: tuple, iterations: int = 100) -> dict:
        # Create dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.inference(dummy_input)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            self.inference(dummy_input)
            times.append((time.time() - start) * 1000)  # Convert to ms

        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times)
        }


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Load model
    model = cfg.model.student
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.device == 'mps':
        device = torch.device('mps')
        model = model.to(device)
    model.eval()

    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 32, 32)
    if args.device == 'mps':
        dummy_input = dummy_input.to(device)

    torch.onnx.export(model, dummy_input, args.output,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Create ONNX Runtime engine
    print("Creating ONNX Runtime engine...")
    engine = ONNXRuntimeEngine(args.output)

    # Test inference
    print("Testing inference...")
    input_data = np.random.random((1, 3, 32, 32)).astype(np.float32)
    output = engine.inference(input_data)
    print(f"Output shape: {output.shape}")

    # Benchmark
    print("Running benchmark...")
    benchmark_results = engine.benchmark(
        input_shape=(1, 3, 32, 32),
        iterations=100
    )
    print("Benchmark results:", benchmark_results)


if __name__ == '__main__':
    main()
