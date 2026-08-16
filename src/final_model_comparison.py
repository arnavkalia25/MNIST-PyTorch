import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from model import MNISTModel
from cnn_model import MNISTCNN


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"

BATCH_SIZE = 64


MLP_MODEL_PATH = "models/best_mnist_mlp.pth"

CNN_MODEL_PATH = "models/best_mnist_cnn.pth"

AUGMENTED_MODEL_PATH = "models/best_mnist_cnn_augmented.pth"

OPTUNA_MODEL_PATH = "models/best_mnist_cnn_optuna.pth"


RESULTS_DIR = "results"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Dataset
# ============================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print()
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(model):

    model.eval()

    correct = 0

    total = 0

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    loss = total_loss / total

    errors = total - correct

    return loss, accuracy, errors


# ============================================================
# Load MLP
# ============================================================

print()
print("Loading MLP...")

mlp = MNISTModel().to(device)

mlp.load_state_dict(
    torch.load(
        MLP_MODEL_PATH,
        map_location=device
    )
)

mlp_loss, mlp_accuracy, mlp_errors = evaluate_model(mlp)

print("✓ MLP evaluated")


# ============================================================
# Load CNN
# ============================================================

print()
print("Loading CNN...")

cnn = MNISTCNN(
    dropout=0.3
).to(device)

cnn.load_state_dict(
    torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )
)

cnn_loss, cnn_accuracy, cnn_errors = evaluate_model(cnn)

print("✓ CNN evaluated")


# ============================================================
# Load CNN + Augmentation
# ============================================================

print()
print("Loading CNN + Augmentation...")

cnn_augmented = MNISTCNN(
    dropout=0.3
).to(device)

cnn_augmented.load_state_dict(
    torch.load(
        AUGMENTED_MODEL_PATH,
        map_location=device
    )
)

augmented_loss, augmented_accuracy, augmented_errors = (
    evaluate_model(cnn_augmented)
)

print("✓ CNN + Augmentation evaluated")


# ============================================================
# Load CNN + Optuna
# ============================================================

print()
print("Loading CNN + Optuna...")

cnn_optuna = MNISTCNN(
    dropout=0.25452660518957637
).to(device)

cnn_optuna.load_state_dict(
    torch.load(
        OPTUNA_MODEL_PATH,
        map_location=device
    )
)

optuna_loss, optuna_accuracy, optuna_errors = (
    evaluate_model(cnn_optuna)
)

print("✓ CNN + Optuna evaluated")


# ============================================================
# Store Results
# ============================================================

models = [
    "MLP",
    "CNN",
    "CNN + Augmentation",
    "CNN + Optuna"
]

accuracies = [
    mlp_accuracy,
    cnn_accuracy,
    augmented_accuracy,
    optuna_accuracy
]

losses = [
    mlp_loss,
    cnn_loss,
    augmented_loss,
    optuna_loss
]

errors = [
    mlp_errors,
    cnn_errors,
    augmented_errors,
    optuna_errors
]


# ============================================================
# Print Comparison
# ============================================================

print()
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print()

print(
    f"{'Model':<25}"
    f"{'Loss':<15}"
    f"{'Accuracy':<15}"
    f"{'Errors':<10}"
)

print("-" * 65)

for i in range(len(models)):

    print(
        f"{models[i]:<25}"
        f"{losses[i]:<15.4f}"
        f"{accuracies[i] * 100:<15.2f}"
        f"{errors[i]:<10}"
    )


# ============================================================
# Improvements
# ============================================================

print()
print("=" * 80)
print("IMPROVEMENTS OVER MLP")
print("=" * 80)

mlp_baseline = mlp_accuracy

for i in range(1, len(models)):

    improvement = (
        accuracies[i] - mlp_baseline
    ) * 100

    print(
        f"{models[i]} vs MLP: "
        f"+{improvement:.2f} percentage points"
    )


# ============================================================
# Find Best Model
# ============================================================

best_index = accuracies.index(
    max(accuracies)
)

best_model = models[best_index]

best_accuracy = accuracies[best_index]


print()
print("=" * 80)
print("BEST MODEL")
print("=" * 80)

print()
print(
    f"Best Model: {best_model}"
)

print(
    f"Test Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Test Errors: "
    f"{errors[best_index]}"
)


# ============================================================
# Create Results Directory
# ============================================================

import os

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Accuracy Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    models,
    [
        accuracy * 100
        for accuracy in accuracies
    ]
)

plt.title(
    "MNIST Model Accuracy Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.ylim(
    95,
    100
)

plt.xticks(
    rotation=15
)

for bar, accuracy in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{accuracy * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

accuracy_path = (
    f"{RESULTS_DIR}/final_model_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ============================================================
# Error Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    models,
    errors
)

plt.title(
    "MNIST Model Error Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Number of Incorrect Predictions"
)

plt.xticks(
    rotation=15
)

for bar, error in zip(
    bars,
    errors
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(error),
        ha="center"
    )

plt.tight_layout()

error_path = (
    f"{RESULTS_DIR}/final_model_errors.png"
)

plt.savefig(
    error_path,
    dpi=300
)

plt.close()


# ============================================================
# Complete
# ============================================================

print()
print("=" * 80)
print("FINAL COMPARISON COMPLETE")
print("=" * 80)

print()
print(
    f"Accuracy chart saved to: "
    f"{accuracy_path}"
)

print(
    f"Error chart saved to: "
    f"{error_path}"
)