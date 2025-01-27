"""Registry module for OpenBlock."""
from mmengine.registry import Registry

# Create registries
DATASETS = Registry('dataset', locations=['openblock.datasets'])
TRANSFORMS = Registry('transform', locations=['openblock.datasets.pipelines'])
SAMPLERS = Registry('sampler', locations=['openblock.datasets.samplers'])
MODELS = Registry('model', locations=['openblock.models'])
VISUALIZERS = Registry('visualizer', locations=['openblock.visualization'])

__all__ = ['DATASETS', 'TRANSFORMS', 'SAMPLERS', 'MODELS', 'VISUALIZERS']
