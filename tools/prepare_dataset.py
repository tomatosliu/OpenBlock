import os
import torchvision
from torchvision import transforms
from mmengine.utils import mkdir_or_exist


def main():
    # Create data directory
    data_root = 'data/cifar10'
    mkdir_or_exist(data_root)

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Download dataset
    print("Downloading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test)

    print(f"Dataset downloaded to {data_root}")
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")


if __name__ == '__main__':
    main()
