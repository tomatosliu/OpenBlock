from typing import Dict, List, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import wandb
from mmengine.visualization import Visualizer
from ..registry import VISUALIZERS


@VISUALIZERS.register_module()
class TrainingMonitor(Visualizer):
    """Visualization tool for monitoring training progress.

    This tool supports:
    - Loss curves visualization
    - Learning rate scheduling visualization
    - Model size and FLOPs comparison
    - Feature map visualization
    - Confusion matrix

    Args:
        name (str): Name of visualization backend
        save_dir (Optional[str]): Directory to save visualizations
        use_wandb (bool): Whether to use Weights & Biases for logging
    """

    def __init__(self,
                 name: str,
                 save_dir: Optional[str] = None,
                 use_wandb: bool = False):
        super().__init__(name)
        self.save_dir = save_dir
        self.use_wandb = use_wandb

        if use_wandb and not wandb.run:
            wandb.init(project=name)

    def plot_losses(self,
                    losses: Dict[str, List[float]],
                    step: Union[int, List[int]],
                    title: str = 'Training Losses') -> Figure:
        """Plot training loss curves.

        Args:
            losses (Dict[str, List[float]]): Dictionary of loss names and values
            step (Union[int, List[int]]): Training step or list of steps
            title (str): Plot title

        Returns:
            Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(step, int):
            step = list(range(step))

        for loss_name, loss_values in losses.items():
            ax.plot(step, loss_values, label=loss_name)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        if self.use_wandb:
            wandb.log({
                'losses': wandb.Image(fig),
                **{k: v[-1] for k, v in losses.items()}
            })

        return fig

    def plot_lr_schedule(self,
                         learning_rates: List[float],
                         step: Union[int, List[int]],
                         title: str = 'Learning Rate Schedule') -> Figure:
        """Plot learning rate schedule.

        Args:
            learning_rates (List[float]): List of learning rates
            step (Union[int, List[int]]): Training step or list of steps
            title (str): Plot title

        Returns:
            Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(step, int):
            step = list(range(step))

        ax.plot(step, learning_rates)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True)

        if self.use_wandb:
            wandb.log({
                'lr_schedule': wandb.Image(fig),
                'current_lr': learning_rates[-1]
            })

        return fig

    def plot_model_comparison(self,
                              model_stats: Dict[str, Dict[str, float]],
                              metrics: List[str] = ['params', 'flops', 'accuracy']) -> Figure:
        """Plot model size and performance comparison.

        Args:
            model_stats (Dict[str, Dict[str, float]]): Dictionary of model statistics
            metrics (List[str]): Metrics to compare

        Returns:
            Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))

        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [stats[metric] for stats in model_stats.values()]
            ax.bar(model_stats.keys(), values)
            ax.set_title(f'Model {metric.capitalize()}')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if self.use_wandb:
            wandb.log({
                'model_comparison': wandb.Image(fig),
                **{f'model_{k}_{m}': v[m]
                   for k, v in model_stats.items()
                   for m in metrics}
            })

        return fig

    def plot_confusion_matrix(self,
                              confusion_matrix: np.ndarray,
                              class_names: List[str],
                              title: str = 'Confusion Matrix') -> Figure:
        """Plot confusion matrix.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix array
            class_names (List[str]): List of class names
            title (str): Plot title

        Returns:
            Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(confusion_matrix, cmap='Blues')

        # Add colorbar
        plt.colorbar(im)

        # Add labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        # Add title
        ax.set_title(title)

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, confusion_matrix[i, j],
                               ha="center", va="center", color="black")

        plt.tight_layout()

        if self.use_wandb:
            wandb.log({
                'confusion_matrix': wandb.Image(fig),
                'accuracy': np.trace(confusion_matrix) / np.sum(confusion_matrix)
            })

        return fig

    def plot_feature_maps(self,
                          feature_maps: torch.Tensor,
                          num_channels: int = 16,
                          title: str = 'Feature Maps') -> Figure:
        """Plot feature map visualizations.

        Args:
            feature_maps (torch.Tensor): Feature maps tensor [C,H,W]
            num_channels (int): Number of channels to visualize
            title (str): Plot title

        Returns:
            Figure: Matplotlib figure
        """
        num_channels = min(num_channels, feature_maps.size(0))
        nrows = int(np.ceil(np.sqrt(num_channels)))
        ncols = int(np.ceil(num_channels / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
        axes = axes.flat

        for i in range(num_channels):
            im = axes[i].imshow(feature_maps[i].cpu(), cmap='viridis')
            axes[i].axis('off')

        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        if self.use_wandb:
            wandb.log({
                'feature_maps': wandb.Image(fig)
            })

        return fig
