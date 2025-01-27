from typing import List, Union, Optional, Dict, Any, Tuple
from enum import Enum
from mmengine.dataset import BaseDataset
from ..registry import DATASETS
import numpy as np


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    LANDMARK = "landmark"


@DATASETS.register_module()
class OpenBlockDataset(BaseDataset):
    """Enhanced base dataset for OpenBlock.

    This dataset is designed to work with MMEngine's training pipeline.
    It supports multiple vision tasks including classification, detection,
    segmentation, and landmark detection.

    Args:
        data_root (str): Root path of the dataset
        ann_file (str): Annotation file path
        data_prefix (Union[str, dict]): Path prefix for data
        pipeline (List[dict]): Processing pipeline
        test_mode (bool): Whether in test mode. Defaults to False
        task_type (str): Type of task from TaskType enum
        class_map (Optional[Dict[str, int]]): Mapping from class names to indices
        sample_ratio (float): Ratio of data to sample, useful for few-shot learning
        cache_mode (bool): Whether to cache data in memory
    """

    METAINFO = {
        'task_type': None,
        'classes': None,
        'num_classes': None,
        'task_specific_params': {}
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Union[str, dict],
                 pipeline: List[dict],
                 test_mode: bool = False,
                 task_type: Optional[str] = None,
                 class_map: Optional[Dict[str, int]] = None,
                 sample_ratio: float = 1.0,
                 cache_mode: bool = False) -> None:

        if task_type is not None:
            try:
                self.task_type = TaskType(task_type)
                self.METAINFO['task_type'] = task_type
            except ValueError:
                raise ValueError(
                    f"Invalid task_type: {task_type}. Must be one of {[t.value for t in TaskType]}")

        self.class_map = class_map
        self.sample_ratio = sample_ratio
        self.cache_mode = cache_mode
        self._data_cache = {} if cache_mode else None

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode)

    def load_data_list(self) -> List[dict]:
        """Load annotation data.

        Returns:
            List[dict]: A list of annotation data.
        """
        raise NotImplementedError

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index with caching support.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.cache_mode and idx in self._data_cache:
            return self._data_cache[idx]

        data_info = super().get_data_info(idx)

        # Apply class mapping if provided
        if self.class_map is not None:
            if self.task_type == TaskType.CLASSIFICATION:
                if 'gt_label' in data_info:
                    label_name = self.METAINFO['classes'][data_info['gt_label']]
                    data_info['gt_label'] = self.class_map.get(
                        label_name, data_info['gt_label'])
            elif self.task_type == TaskType.DETECTION:
                if 'gt_bboxes_labels' in data_info:
                    labels = data_info['gt_bboxes_labels']
                    mapped_labels = [self.class_map.get(
                        self.METAINFO['classes'][l], l) for l in labels]
                    data_info['gt_bboxes_labels'] = np.array(
                        mapped_labels, dtype=np.int64)

        if self.cache_mode:
            self._data_cache[idx] = data_info

        return data_info

    def prepare_data(self, idx: int) -> Union[dict, tuple]:
        """Get data processed by pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            Union[dict, tuple]: Processed data.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def get_subset(self, indices: List[int]) -> 'OpenBlockDataset':
        """Create a subset of the dataset.

        Args:
            indices (List[int]): Indices to include in subset

        Returns:
            OpenBlockDataset: A new dataset instance containing only the specified indices
        """
        subset = type(self)(
            data_root=self.data_root,
            ann_file=self.ann_file,
            data_prefix=self.data_prefix,
            pipeline=self.pipeline,
            test_mode=self.test_mode,
            task_type=self.task_type.value if self.task_type else None,
            class_map=self.class_map,
            sample_ratio=1.0,  # No need to sample again
            cache_mode=self.cache_mode
        )
        subset.data_list = [self.data_list[i] for i in indices]
        return subset

    @property
    def task_specific_params(self) -> Dict[str, Any]:
        """Get task-specific parameters.

        Returns:
            Dict[str, Any]: Task-specific parameters
        """
        return self.METAINFO.get('task_specific_params', {})

    def set_task_specific_params(self, params: Dict[str, Any]) -> None:
        """Set task-specific parameters.

        Args:
            params (Dict[str, Any]): Parameters to set
        """
        self.METAINFO['task_specific_params'] = params
