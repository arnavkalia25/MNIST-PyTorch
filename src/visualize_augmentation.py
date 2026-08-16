
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"


# ============================================================
# Data Augmentation Transform
# ============================================================

augmentation_transform = transforms.Compose([

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
# Load MNIST Dataset
# ============================================================

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True
)


# ============================================================
# Select one image
# ============================================================

index = 0

image, label = dataset[index]


# ============================================================
# Create augmented versions
# ============================================================

augmented_images = []

for _ in range(5):

    augmented_image = augmentation_transform(
        image
    )

    augmented_images.append(
        augmented_image
    )


# ============================================================
# Plot Original Image
# ============================================================

plt.figure(figsize=(12, 3))

plt.subplot(1, 6, 1)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    f"Original\nLabel: {label}"
)

plt.axis("off")


# ============================================================
# Plot Augmented Images
# ============================================================

for i, augmented_image in enumerate(
    augmented_images
):

    plt.subplot(
        1,
        6,
        i + 2
    )

    plt.imshow(
        augmented_image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Augmented {i + 1}"
    )

    plt.axis("off")


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save Figure
# ============================================================

plt.savefig(
    "results/augmentation_examples.png",
    dpi=150,
    bbox_inches="tight"
)


# ============================================================
# Display
# ============================================================

plt.show()

print()
print("===== DATA AUGMENTATION VISUALIZATION =====")
print()
print("Original label:", label)
print()
print("Generated 5 augmented versions.")
print()
print("Saved:")
print("results/augmentation_examples.png")

