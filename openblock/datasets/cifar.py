from typing import List
import os.path as osp
import torchvision
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class CIFAR10Dataset(BaseDataset):
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
        'task_type': 'classification',
    }

    def __init__(self,
                 data_root: str,
                 data_prefix: dict,
                 test_mode: bool = False,
                 pipeline: List[dict] = None):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        """Load annotation data.

        Returns:
            List[dict]: A list of annotation data
        """
        # Load CIFAR10 using torchvision
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=not self.test_mode,
            download=False)  # Already downloaded

        data_list = []
        for idx, (img, label) in enumerate(dataset):
            info = {
                'img_id': idx,
                'img_path': osp.join(self.data_root, f'img_{idx}.png'),
                'gt_label': int(label),
            }
            data_list.append(info)

        return data_list
