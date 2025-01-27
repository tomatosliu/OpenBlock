from typing import List, Sequence
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


@METRICS.register_module()
class Accuracy(BaseMetric):
    """Accuracy evaluation metric.

    Args:
        topk (Sequence[int]): The k values to calculate top-k accuracy.
            Defaults to (1, ).
        collect_device (str): Device name used for collecting results from different
            ranks during distributed training. Must be 'cpu' or 'gpu'.
            Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric names to
            disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 topk: Sequence[int] = (1, ),
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device, prefix)
        self.topk = topk

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_label']
            label = data_sample['gt_label']
            result['pred_label'] = pred
            result['gt_label'] = label
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        predictions = torch.tensor([res['pred_label'] for res in results])
        labels = torch.tensor([res['gt_label'] for res in results])

        metrics = {}
        for k in self.topk:
            top_k_correct = self._top_k_accuracy(predictions, labels, k)
            metrics[f'top{k}_acc'] = top_k_correct.item() * 100

        return metrics

    def _top_k_accuracy(self, pred, target, k):
        """Calculate top-k accuracy.

        Args:
            pred (torch.Tensor): Prediction labels.
            target (torch.Tensor): Ground truth labels.
            k (int): K value for top-k accuracy.

        Returns:
            torch.Tensor: Top-k accuracy.
        """
        with torch.no_grad():
            if pred.shape != target.shape:
                # Handle the case where pred is logits
                pred = torch.argmax(pred, dim=1)
            correct = pred.eq(target)
            return correct.float().mean()
