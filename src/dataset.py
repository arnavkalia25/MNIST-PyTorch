
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64

DATA_DIR = "data"


# ============================================================
# Training Transform
# ============================================================
#
# These transformations are applied ONLY to training images.
#
# RandomRotation:
#   Rotates the digit by up to +/- 10 degrees.
#
# RandomAffine:
#   Moves the digit slightly horizontally and vertically.
#
# ToTensor:
#   Converts the image to a PyTorch tensor.
#
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10)
    ),

    transforms.ToTensor()
])


# ============================================================
# Validation / Test Transform
# ============================================================
#
# Validation and test images must NOT be randomly modified.
#
# ============================================================

test_transform = transforms.ToTensor()


# ============================================================
# Get DataLoaders
# ============================================================

def get_dataloaders():

    # ========================================================
    # Create the original MNIST training dataset
    # ========================================================
    #
    # We initially use transform=None because we only want
    # the dataset to provide the images and labels.
    #
    # ========================================================

    full_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )

    # ========================================================
    # Test dataset
    # ========================================================

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    # ========================================================
    # Create reproducible train/validation split
    # ========================================================

    train_size = int(
        0.8 * len(full_dataset)
    )

    validation_size = (
        len(full_dataset)
        - train_size
    )

    train_subset, validation_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ========================================================
    # Get the exact indices
    # ========================================================

    train_indices = train_subset.indices

    validation_indices = validation_subset.indices

    # ========================================================
    # Create separate datasets
    # ========================================================
    #
    # IMPORTANT:
    #
    # Training dataset:
    #     RandomRotation
    #     RandomAffine
    #     ToTensor
    #
    # Validation dataset:
    #     ToTensor only
    #
    # This prevents data augmentation from affecting
    # validation performance.
    #
    # ========================================================

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    validation_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=test_transform
    )

    # ========================================================
    # Apply the same split indices
    # ========================================================

    train_dataset = Subset(
        train_dataset,
        train_indices
    )

    validation_dataset = Subset(
        validation_dataset,
        validation_indices
    )

    # ========================================================
    # Training DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ========================================================
    # Validation DataLoader
    # ========================================================

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # Test DataLoader
    # ========================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # Return all three
    # ========================================================

    return (
        train_loader,
        validation_loader,
        test_loader
    )

