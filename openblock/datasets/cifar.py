import os.path as osp
import torchvision
import torch
import numpy as np
from typing import Dict, List, Optional

from .base_dataset import OpenBlockDataset, TaskType
from ..registry import DATASETS


@DATASETS.register_module()
class CIFAR10Dataset(OpenBlockDataset):
    """CIFAR10 Dataset for OpenBlock.

    Args:
        data_root (str): Dataset root directory
        data_prefix (dict): Path prefix for data
        test_mode (bool): Whether in test mode. Defaults to False
        pipeline (List[dict]): Processing pipeline
    """

    METAINFO = {
        'classes': ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'),
        'task_type': TaskType.CLASSIFICATION.value,
        'num_classes': 10
    }

    def __init__(self,
                 data_root: str,
                 data_prefix: dict,
                 test_mode: bool = False,
                 pipeline: List[dict] = None):
        # Initialize dataset first
        self._dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=not test_mode,
            download=True)  # Auto download if not exists

        # CIFAR10 doesn't need annotation file
        super().__init__(
            data_root=data_root,
            ann_file='',  # Empty string as we don't need annotation file
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            task_type=TaskType.CLASSIFICATION.value)

    def load_data_list(self) -> List[dict]:
        """Load annotation data.

        Returns:
            List[dict]: A list of annotation data
        """
        data_list = []
        for idx, (img, label) in enumerate(self._dataset):
            info = {
                'img_id': idx,
                'img': np.array(img),  # Convert PIL Image to numpy array
                'gt_label': int(label),
            }
            data_list.append(info)

        return data_list
