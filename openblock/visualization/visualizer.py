from typing import Optional, Union, Sequence
import numpy as np
import torch
from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS
from mmengine.structures import BaseDataElement


@VISUALIZERS.register_module()
class OpenBlockVisualizer(Visualizer):
    """Visualizer for OpenBlock.

    This visualizer extends MMEngine's Visualizer to provide visualization
    for model optimization results, training curves, and model comparison.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save directory. Defaults to None.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Sequence[dict]] = None,
                 save_dir: Optional[str] = None) -> None:
        super().__init__(name, image, vis_backends, save_dir)

    def add_model_comparison(self,
                             name: str,
                             original_outputs: Union[torch.Tensor, np.ndarray],
                             optimized_outputs: Union[torch.Tensor, np.ndarray],
                             step: int = 0) -> None:
        """Add model comparison results.

        Args:
            name (str): The visualization identifier
            original_outputs: Outputs from original model
            optimized_outputs: Outputs from optimized model
            step (int): Global step value. Defaults to 0
        """
        if isinstance(original_outputs, torch.Tensor):
            original_outputs = original_outputs.detach().cpu().numpy()
        if isinstance(optimized_outputs, torch.Tensor):
            optimized_outputs = optimized_outputs.detach().cpu().numpy()

        # Calculate differences
        diff = np.abs(original_outputs - optimized_outputs)

        # Add to tensorboard/wandb
        self.add_scalar(f'{name}/max_diff', np.max(diff), step)
        self.add_scalar(f'{name}/mean_diff', np.mean(diff), step)
        self.add_scalar(f'{name}/std_diff', np.std(diff), step)

    def add_optimization_metrics(self,
                                 name: str,
                                 metrics: dict,
                                 step: int = 0) -> None:
        """Add optimization metrics.

        Args:
            name (str): The visualization identifier
            metrics (dict): Dictionary of metrics
            step (int): Global step value. Defaults to 0
        """
        for metric_name, value in metrics.items():
            self.add_scalar(f'{name}/{metric_name}', value, step)

    def visualize_distillation(self,
                               name: str,
                               teacher_feat: Union[torch.Tensor, np.ndarray],
                               student_feat: Union[torch.Tensor, np.ndarray],
                               step: int = 0) -> None:
        """Visualize feature maps from teacher and student models.

        Args:
            name (str): The visualization identifier
            teacher_feat: Feature maps from teacher model
            student_feat: Feature maps from student model
            step (int): Global step value. Defaults to 0
        """
        if isinstance(teacher_feat, torch.Tensor):
            teacher_feat = teacher_feat.detach().cpu().numpy()
        if isinstance(student_feat, torch.Tensor):
            student_feat = student_feat.detach().cpu().numpy()

        # Add feature map visualizations
        self.add_image(f'{name}/teacher', teacher_feat, step)
        self.add_image(f'{name}/student', student_feat, step)

        # Add difference map
        diff_map = np.abs(teacher_feat - student_feat)
        self.add_image(f'{name}/diff_map', diff_map, step)

    def plot_model_structure(self,
                             model: torch.nn.Module,
                             save_path: Optional[str] = None) -> None:
        """Plot model structure.

        Args:
            model (torch.nn.Module): The model to visualize
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        try:
            from torchviz import make_dot

            # Create dummy input
            x = torch.randn(1, 3, 224, 224)
            y = model(x)

            # Generate computational graph
            dot = make_dot(y, params=dict(model.named_parameters()))

            if save_path is not None:
                dot.render(save_path, format='png')

        except ImportError:
            print("Please install graphviz and torchviz to plot model structure")
