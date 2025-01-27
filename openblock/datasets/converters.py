from typing import Dict, List, Optional, Union
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from .base_dataset import TaskType


class DatasetConverter:
    """Utility class for converting between different dataset formats.

    Supports conversion between:
    - COCO
    - VOC
    - YOLO
    - Custom formats
    """

    @staticmethod
    def coco_to_openblock(
        coco_file: str,
        img_dir: str,
        task_type: TaskType,
        class_map: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """Convert COCO format annotations to OpenBlock format.

        Args:
            coco_file (str): Path to COCO annotation JSON file
            img_dir (str): Directory containing images
            task_type (TaskType): Type of task (detection/segmentation)
            class_map (Optional[Dict[str, int]]): Optional class name to index mapping

        Returns:
            List[Dict]: List of OpenBlock format annotations
        """
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Build category id to name mapping
        cat_id_to_name = {cat['id']: cat['name']
                          for cat in coco_data['categories']}

        # Build image id to file name mapping
        img_id_to_file = {img['id']: img['file_name']
                          for img in coco_data['images']}

        results = []
        for img in coco_data['images']:
            img_id = img['id']
            img_anns = [ann for ann in coco_data['annotations']
                        if ann['image_id'] == img_id]

            data_info = {
                'img_id': img_id,
                'img_path': os.path.join(img_dir, img_id_to_file[img_id]),
                'height': img['height'],
                'width': img['width']
            }

            if task_type == TaskType.DETECTION:
                bboxes = []
                labels = []
                for ann in img_anns:
                    bbox = ann['bbox']  # [x,y,w,h]
                    # Convert to [x1,y1,x2,y2]
                    bbox = [
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3]
                    ]
                    bboxes.append(bbox)

                    cat_name = cat_id_to_name[ann['category_id']]
                    label = class_map[cat_name] if class_map else ann['category_id']
                    labels.append(label)

                data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
                data_info['gt_bboxes_labels'] = np.array(
                    labels, dtype=np.int64)

            elif task_type == TaskType.SEGMENTATION:
                masks = []
                labels = []
                for ann in img_anns:
                    mask = ann['segmentation']  # RLE or polygon
                    masks.append(mask)

                    cat_name = cat_id_to_name[ann['category_id']]
                    label = class_map[cat_name] if class_map else ann['category_id']
                    labels.append(label)

                data_info['gt_masks'] = masks
                data_info['gt_masks_labels'] = np.array(labels, dtype=np.int64)

            results.append(data_info)

        return results

    @staticmethod
    def voc_to_openblock(
        voc_dir: str,
        img_dir: str,
        class_map: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """Convert VOC format annotations to OpenBlock format.

        Args:
            voc_dir (str): Directory containing VOC XML annotations
            img_dir (str): Directory containing images
            class_map (Optional[Dict[str, int]]): Optional class name to index mapping

        Returns:
            List[Dict]: List of OpenBlock format annotations
        """
        results = []
        for xml_file in Path(voc_dir).glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_file = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            data_info = {
                'img_id': img_file,
                'img_path': os.path.join(img_dir, img_file),
                'height': height,
                'width': width
            }

            bboxes = []
            labels = []

            for obj in root.findall('object'):
                name = obj.find('name').text
                label = class_map[name] if class_map else name

                bbox = obj.find('bndbox')
                bbox = [
                    float(bbox.find('xmin').text),
                    float(bbox.find('ymin').text),
                    float(bbox.find('xmax').text),
                    float(bbox.find('ymax').text)
                ]

                bboxes.append(bbox)
                labels.append(label)

            if bboxes:
                data_info['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
                data_info['gt_bboxes_labels'] = np.array(
                    labels, dtype=np.int64)

            results.append(data_info)

        return results

    @staticmethod
    def classification_to_openblock(
        img_dir: str,
        class_map: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """Convert classification dataset to OpenBlock format.

        Assumes images are organized in class folders:
        img_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...

        Args:
            img_dir (str): Root directory containing class folders
            class_map (Optional[Dict[str, int]]): Optional class name to index mapping

        Returns:
            List[Dict]: List of OpenBlock format annotations
        """
        results = []
        img_id = 0

        for class_name in sorted(os.listdir(img_dir)):
            class_dir = os.path.join(img_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = class_map[class_name] if class_map else len(results)

            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(class_dir, img_file)
                with Image.open(img_path) as img:
                    width, height = img.size

                data_info = {
                    'img_id': img_id,
                    'img_path': img_path,
                    'gt_label': label,
                    'height': height,
                    'width': width
                }

                results.append(data_info)
                img_id += 1

        return results
