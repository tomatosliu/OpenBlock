from typing import List, Dict, Optional
import numpy as np
import json
import os.path as osp
from mmengine.registry import DATASETS

from .base_dataset import OpenBlockDataset, TaskType


@DATASETS.register_module()
class LandmarkDataset(OpenBlockDataset):
    """Dataset for landmark/keypoint detection tasks.

    This dataset supports both 2D and 3D landmarks, with optional visibility flags.
    The annotation format follows a simplified version of COCO keypoints format.

    Args:
        data_root (str): Dataset root directory
        ann_file (str): Annotation file path
        data_prefix (dict): Path prefix for data
        pipeline (List[dict]): Processing pipeline
        num_landmarks (int): Number of landmarks per instance
        landmark_dims (int): Dimensions of landmarks (2 or 3)
        test_mode (bool): Whether in test mode
        class_map (Optional[Dict[str, int]]): Class name to index mapping
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: dict,
                 pipeline: List[dict],
                 num_landmarks: int,
                 landmark_dims: int = 2,
                 test_mode: bool = False,
                 class_map: Optional[Dict[str, int]] = None):
        self.num_landmarks = num_landmarks
        self.landmark_dims = landmark_dims
        assert landmark_dims in [
            2, 3], f'Invalid landmark_dims: {landmark_dims}'

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            task_type=TaskType.LANDMARK.value,
            class_map=class_map)

        # Set task-specific parameters
        self.set_task_specific_params({
            'num_landmarks': num_landmarks,
            'landmark_dims': landmark_dims
        })

    def load_data_list(self) -> List[dict]:
        """Load landmark annotation data.

        The annotation file should be a JSON file with the following format:
        {
            "images": [
                {
                    "id": int,
                    "file_name": str,
                    "height": int,
                    "width": int
                }
            ],
            "annotations": [
                {
                    "image_id": int,
                    "category_id": int,
                    "landmarks": List[float],  # [x1,y1,v1,x2,y2,v2,...] or [x1,y1,z1,v1,...]
                    "bbox": [x,y,w,h]  # optional
                }
            ],
            "categories": [
                {
                    "id": int,
                    "name": str
                }
            ]
        }

        Returns:
            List[dict]: List of annotation data
        """
        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        # Build mappings
        img_id_to_file = {img['id']: img['file_name']
                          for img in data['images']}
        if 'categories' in data:
            cat_id_to_name = {cat['id']: cat['name']
                              for cat in data['categories']}

        results = []
        for img in data['images']:
            img_id = img['id']
            img_anns = [ann for ann in data['annotations']
                        if ann['image_id'] == img_id]

            data_info = {
                'img_id': img_id,
                'img_path': osp.join(self.data_prefix['img'], img_id_to_file[img_id]),
                'height': img['height'],
                'width': img['width']
            }

            if img_anns:
                landmarks = []
                visibilities = []
                labels = []
                bboxes = []

                for ann in img_anns:
                    # Process landmarks
                    lm = np.array(ann['landmarks'])
                    if self.landmark_dims == 2:
                        # Format: [x1,y1,v1,x2,y2,v2,...]
                        lm = lm.reshape(-1, 3)
                        landmarks.append(lm[:, :2])
                        visibilities.append(lm[:, 2])
                    else:  # 3D
                        # Format: [x1,y1,z1,v1,x2,y2,z2,v2,...]
                        lm = lm.reshape(-1, 4)
                        landmarks.append(lm[:, :3])
                        visibilities.append(lm[:, 3])

                    # Process category
                    if 'category_id' in ann:
                        cat_name = cat_id_to_name[ann['category_id']]
                        label = self.class_map[cat_name] if self.class_map else ann['category_id']
                        labels.append(label)

                    # Process bbox if available
                    if 'bbox' in ann:
                        bbox = ann['bbox']  # [x,y,w,h]
                        bbox = [
                            bbox[0],
                            bbox[1],
                            bbox[0] + bbox[2],
                            bbox[1] + bbox[3]
                        ]
                        bboxes.append(bbox)

                data_info['gt_landmarks'] = np.array(
                    landmarks, dtype=np.float32)
                data_info['gt_landmarks_visibility'] = np.array(
                    visibilities, dtype=np.float32)

                if labels:
                    data_info['gt_labels'] = np.array(labels, dtype=np.int64)
                if bboxes:
                    data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)

            results.append(data_info)

        return results
