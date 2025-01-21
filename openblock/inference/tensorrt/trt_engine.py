import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Dict, Any, Union, List
import torch

from ..base_engine import BaseEngine


class TRTEngine(BaseEngine):
    """TensorRT inference engine"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.context = None
        self.engine = None
        self.stream = None
        self.host_inputs = None
        self.host_outputs = None
        self.device_inputs = None
        self.device_outputs = None
        self.bindings = None

    def load(self):
        """Load TensorRT model"""
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Allocate memory
        self.stream = cuda.Stream()
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def preprocess(self, input_data: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Preprocess input data"""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        # Add preprocessing logic, e.g., normalization, resize, etc.
        processed_data = input_data.ravel()
        np.copyto(self.host_inputs[0], processed_data)
        return [processed_data]

    def inference(self, input_data: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        """Run inference"""
        self.preprocess(input_data)

        # Copy input data to GPU
        for host_input, device_input in zip(self.host_inputs, self.device_inputs):
            cuda.memcpy_htod_async(device_input, host_input, self.stream)

        # Execute inference
        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # Copy output data to CPU
        outputs = []
        for host_output, device_output in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            outputs.append(host_output)

        self.stream.synchronize()
        return self.postprocess(outputs)

    def postprocess(self, outputs: List[np.ndarray]) -> Any:
        """Post-process results"""
        # Add post-processing logic, e.g., softmax, etc.
        return outputs

    def benchmark(self, input_shape: tuple, iterations: int = 100) -> Dict[str, float]:
        """Run performance benchmark"""
        dummy_input = np.random.random(input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.inference(dummy_input)

        # Test latency
        start = cuda.Event()
        end = cuda.Event()
        times = []

        for _ in range(iterations):
            start.record()
            self.inference(dummy_input)
            end.record()
            end.synchronize()
            times.append(start.time_till(end))

        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times)
        }

    def release(self):
        """Release resources"""
        for device_input in self.device_inputs:
            device_input.free()
        for device_output in self.device_outputs:
            device_output.free()
        self.stream.free()
