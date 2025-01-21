from typing import List, Union, Optional
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class OpenBlockDataset(BaseDataset):
    """Base dataset for OpenBlock.

    This dataset is designed to work with MMEngine's training pipeline.
    It supports both classification and detection tasks.

    Args:
        data_root (str): Root path of the dataset
        ann_file (str): Annotation file path
        data_prefix (Union[str, dict]): Path prefix for data
        pipeline (List[dict]): Processing pipeline
        test_mode (bool): Whether in test mode. Defaults to False
        task_type (str): Type of task, e.g., 'classification', 'detection'
    """

    METAINFO = {
        'task_type': None,
        'classes': None
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Union[str, dict],
                 pipeline: List[dict],
                 test_mode: bool = False,
                 task_type: Optional[str] = None) -> None:
        self.task_type = task_type
        if task_type is not None:
            self.METAINFO['task_type'] = task_type

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
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        data_info = super().get_data_info(idx)
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
