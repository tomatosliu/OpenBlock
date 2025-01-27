from typing import Iterator, List, Optional, Set
import numpy as np
from torch.utils.data import Sampler
import torch
from ..registry import SAMPLERS


@SAMPLERS.register_module()
class BalancedClassSampler(Sampler):
    """Sampler that ensures balanced class distribution in each batch.

    This is particularly useful for imbalanced datasets and few-shot learning scenarios.

    Args:
        dataset: Dataset to sample from
        num_samples (int): Number of samples to draw
        num_classes (int): Number of classes
        labels (List[int]): List of class labels for each sample
        replacement (bool): Whether to sample with replacement
        class_weights (Optional[List[float]]): Weight for each class
    """

    def __init__(self,
                 dataset,
                 num_samples: int,
                 num_classes: int,
                 labels: List[int],
                 replacement: bool = True,
                 class_weights: Optional[List[float]] = None):
        self.dataset = dataset
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.labels = np.array(labels)
        self.replacement = replacement
        self.class_weights = class_weights

        # Build class indices
        self.class_indices = [
            np.where(self.labels == i)[0] for i in range(self.num_classes)
        ]

        # Compute sampling weights for each class
        if class_weights is None:
            # Default to balanced weighting
            class_weights = [1.0 / self.num_classes] * self.num_classes
        else:
            total = sum(class_weights)
            class_weights = [w / total for w in class_weights]

        self.class_weights = np.array(class_weights)

    def __iter__(self) -> Iterator[int]:
        # Determine number of samples per class
        samples_per_class = np.floor(
            self.num_samples * self.class_weights).astype(int)
        remainder = self.num_samples - sum(samples_per_class)

        # Distribute remainder randomly according to weights
        if remainder > 0:
            additional = np.random.choice(
                self.num_classes,
                size=remainder,
                p=self.class_weights
            )
            for cls_idx in additional:
                samples_per_class[cls_idx] += 1

        # Sample indices for each class
        indices = []
        for cls_idx, n_samples in enumerate(samples_per_class):
            if n_samples == 0:
                continue

            cls_indices = self.class_indices[cls_idx]
            if len(cls_indices) == 0:
                continue

            if self.replacement:
                sampled = np.random.choice(
                    cls_indices,
                    size=n_samples,
                    replace=True
                )
            else:
                # If not enough samples, cycle through the available ones
                if n_samples > len(cls_indices):
                    num_full_repeats = n_samples // len(cls_indices)
                    remainder = n_samples % len(cls_indices)
                    sampled = np.concatenate([
                        np.repeat(cls_indices, num_full_repeats),
                        np.random.choice(
                            cls_indices, size=remainder, replace=False)
                    ])
                else:
                    sampled = np.random.choice(
                        cls_indices,
                        size=n_samples,
                        replace=False
                    )

            indices.extend(sampled.tolist())

        # Shuffle the final list of indices
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


@SAMPLERS.register_module()
class DiversityBasedSampler(Sampler):
    """Sampler that promotes diversity in the selected samples.

    This sampler is useful for active learning and few-shot scenarios where we want
    to maximize the diversity of the selected samples.

    Args:
        dataset: Dataset to sample from
        num_samples (int): Number of samples to draw
        features (torch.Tensor): Feature vectors for each sample
        distance_threshold (float): Minimum distance between selected samples
        selection_method (str): Method to select diverse samples ('greedy' or 'kmeans')
    """

    def __init__(self,
                 dataset,
                 num_samples: int,
                 features: torch.Tensor,
                 distance_threshold: float = 0.5,
                 selection_method: str = 'greedy'):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.features = features
        self.distance_threshold = distance_threshold
        self.selection_method = selection_method

        # Normalize features
        self.features = self.features / \
            torch.norm(self.features, dim=1, keepdim=True)

        # Pre-compute diverse sample indices
        self.indices = self._select_diverse_samples()

    def _select_diverse_samples(self) -> List[int]:
        if self.selection_method == 'greedy':
            return self._greedy_selection()
        elif self.selection_method == 'kmeans':
            return self._kmeans_selection()
        else:
            raise ValueError(
                f"Unknown selection method: {self.selection_method}")

    def _greedy_selection(self) -> List[int]:
        """Greedily select diverse samples."""
        selected_indices: Set[int] = set()
        all_indices = set(range(len(self.dataset)))

        # Start with a random sample
        current_idx = np.random.choice(len(self.dataset))
        selected_indices.add(current_idx)

        while len(selected_indices) < self.num_samples:
            remaining = list(all_indices - selected_indices)
            if not remaining:
                break

            # Compute distances to all selected samples
            current_features = self.features[list(selected_indices)]
            remaining_features = self.features[remaining]

            # Compute minimum distance to any selected sample
            distances = torch.mm(remaining_features, current_features.t())
            min_distances, _ = torch.max(distances, dim=1)

            # Select the sample with maximum minimum distance
            max_idx = torch.argmin(min_distances)
            selected_indices.add(remaining[max_idx])

        return list(selected_indices)

    def _kmeans_selection(self) -> List[int]:
        """Select diverse samples using k-means clustering."""
        from sklearn.cluster import KMeans

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.num_samples, random_state=42)
        kmeans.fit(self.features.numpy())

        # Select samples closest to cluster centers
        centers = torch.tensor(kmeans.cluster_centers_)
        distances = torch.cdist(self.features, centers)
        selected_indices = []

        for cluster_idx in range(self.num_samples):
            cluster_samples = torch.where(kmeans.labels_ == cluster_idx)[0]
            if len(cluster_samples) > 0:
                # Select sample closest to cluster center
                cluster_distances = distances[cluster_samples, cluster_idx]
                selected_idx = cluster_samples[torch.argmin(cluster_distances)]
                selected_indices.append(selected_idx.item())

        return selected_indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
