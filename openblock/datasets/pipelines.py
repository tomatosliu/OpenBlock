from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from mmengine.structures import InstanceData, LabelData
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """Load an image from file."""

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): Result dict containing image data.

        Returns:
            dict: Result dict containing loaded image.
        """
        # For CIFAR10, image is already loaded as numpy array
        if isinstance(results['img'], np.ndarray):
            return results

        # For other cases where we need to load from file
        img = results['img']
        if isinstance(img, str):
            img = Image.open(img)
            img = np.array(img)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class ImageClassificationTransform:
    """Apply transforms to image for classification."""

    def __init__(self, size=224, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), is_train=True):
        self.size = size
        self.mean = mean
        self.std = std
        self.is_train = is_train

        # Define transforms
        transforms = []
        transforms.append(T.ToPILImage())
        if self.is_train:
            transforms.extend([
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
            ])
        else:
            transforms.extend([
                T.Resize(int(size * 1.14)),
                T.CenterCrop(size),
            ])

        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        self.transforms = T.Compose(transforms)

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): Result dict containing image data.

        Returns:
            dict: Result dict with transformed image.
        """
        img = results['img']
        img = self.transforms(img)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class PackClassificationInputs:
    """Pack the inputs data for classification."""

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): Result dict containing image data and annotations.

        Returns:
            dict: Formatted dict with inputs and data_samples.
        """
        packed_results = dict()
        # Pack image tensor
        packed_results['inputs'] = results['img']

        # Pack data samples
        data_sample = LabelData()
        data_sample.set_data({
            'gt_label': results.get('gt_label', None),
            'img_id': results.get('img_id', None)
        })

        packed_results['data_samples'] = [data_sample]
        return packed_results


@TRANSFORMS.register_module()
class TeacherStudentTransform:
    """Transform pipeline for teacher-student knowledge distillation.

    This pipeline applies different augmentations to the same image
    for teacher and student models.

    Args:
        teacher_transform: Transform pipeline for teacher model
        student_transform: Transform pipeline for student model
        strong_aug (bool): Whether to use strong augmentation for student
    """

    def __init__(self,
                 teacher_transform: ImageClassificationTransform,
                 student_transform: Optional[ImageClassificationTransform] = None,
                 strong_aug: bool = True):
        self.teacher_transform = teacher_transform

        if student_transform is None:
            # Create stronger augmentation for student by default
            aug_params = {'random_rotation': True,
                          'random_erasing': True} if strong_aug else None
            self.student_transform = ImageClassificationTransform(
                is_train=True,
                aug_params=aug_params
            )
        else:
            self.student_transform = student_transform

    def __call__(self, results: Dict) -> Dict:
        """Apply transforms to the input data.

        Args:
            results (Dict): Data dict containing image and annotations

        Returns:
            Dict: Transformed data with both teacher and student versions
        """
        # Create a copy of the original results
        teacher_results = results.copy()
        student_results = results.copy()

        # Apply transforms
        teacher_results = self.teacher_transform(teacher_results)
        student_results = self.student_transform(student_results)

        # Combine results
        results['teacher_img'] = teacher_results['img']
        results['student_img'] = student_results['img']

        return results
