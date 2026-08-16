
# ============================================================
# MNIST Model Comparison
#
# MLP vs CNN vs CNN + Data Augmentation
# ============================================================

import torch
import torch.nn as nn

from model import MNISTModel
from cnn_model import MNISTCNN
from dataset import get_dataloaders


# ============================================================
# Configuration
# ============================================================

MLP_MODEL_PATH = "models/best_mnist_mlp.pth"

CNN_MODEL_PATH = "models/best_mnist_cnn.pth"

AUGMENTED_CNN_MODEL_PATH = (
    "models/best_mnist_cnn_augmented.pth"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    model,
    test_loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

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

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    average_loss = total_loss / total

    accuracy = (
        100.0 * correct / total
    )

    errors = total - correct

    return (
        average_loss,
        accuracy,
        correct,
        errors
    )


# ============================================================
# Load MLP
# ============================================================

def load_mlp():

    model = MNISTModel().to(device)

    checkpoint = torch.load(
        MLP_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# Load CNN
# ============================================================

def load_cnn():

    model = MNISTCNN().to(device)

    checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# Load Augmented CNN
# ============================================================

def load_augmented_cnn():

    model = MNISTCNN().to(device)

    checkpoint = torch.load(
        AUGMENTED_CNN_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load Dataset
    # ========================================================

    (
        train_loader,
        validation_loader,
        test_loader
    ) = get_dataloaders()

    print()

    print(
        "Test samples:",
        len(test_loader.dataset)
    )

    # ========================================================
    # Loss Function
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Load Models
    # ========================================================

    print()
    print("Loading models...")

    mlp = load_mlp()

    print("✓ MLP loaded")

    cnn = load_cnn()

    print("✓ CNN loaded")

    augmented_cnn = load_augmented_cnn()

    print("✓ Augmented CNN loaded")

    # ========================================================
    # Evaluate MLP
    # ========================================================

    print()
    print("Evaluating MLP...")

    mlp_loss, mlp_accuracy, mlp_correct, mlp_errors = (
        evaluate_model(
            mlp,
            test_loader,
            criterion
        )
    )

    # ========================================================
    # Evaluate CNN
    # ========================================================

    print("Evaluating CNN...")

    cnn_loss, cnn_accuracy, cnn_correct, cnn_errors = (
        evaluate_model(
            cnn,
            test_loader,
            criterion
        )
    )

    # ========================================================
    # Evaluate Augmented CNN
    # ========================================================

    print("Evaluating CNN + Augmentation...")

    aug_loss, aug_accuracy, aug_correct, aug_errors = (
        evaluate_model(
            augmented_cnn,
            test_loader,
            criterion
        )
    )

    # ========================================================
    # Comparison Table
    # ========================================================

    print()
    print("=" * 75)

    print(
        "                 MNIST MODEL COMPARISON"
    )

    print("=" * 75)

    print()

    print(
        f"{'Model':<28}"
        f"{'Loss':>12}"
        f"{'Accuracy':>15}"
        f"{'Errors':>12}"
    )

    print("-" * 75)

    print(
        f"{'MLP':<28}"
        f"{mlp_loss:>12.4f}"
        f"{mlp_accuracy:>14.2f}%"
        f"{mlp_errors:>12}"
    )

    print(
        f"{'CNN':<28}"
        f"{cnn_loss:>12.4f}"
        f"{cnn_accuracy:>14.2f}%"
        f"{cnn_errors:>12}"
    )

    print(
        f"{'CNN + Augmentation':<28}"
        f"{aug_loss:>12.4f}"
        f"{aug_accuracy:>14.2f}%"
        f"{aug_errors:>12}"
    )

    print("-" * 75)

    # ========================================================
    # Improvements
    # ========================================================

    print()

    print(
        "===== IMPROVEMENTS ====="
    )

    print()

    cnn_vs_mlp = (
        cnn_accuracy - mlp_accuracy
    )

    aug_vs_mlp = (
        aug_accuracy - mlp_accuracy
    )

    aug_vs_cnn = (
        aug_accuracy - cnn_accuracy
    )

    print(
        f"CNN vs MLP: "
        f"{cnn_vs_mlp:+.2f} percentage points"
    )

    print(
        f"CNN + Augmentation vs MLP: "
        f"{aug_vs_mlp:+.2f} percentage points"
    )

    print(
        f"CNN + Augmentation vs CNN: "
        f"{aug_vs_cnn:+.2f} percentage points"
    )

    # ========================================================
    # Best Model
    # ========================================================

    results = {

        "MLP": mlp_accuracy,

        "CNN": cnn_accuracy,

        "CNN + Augmentation": aug_accuracy

    }

    best_model = max(
        results,
        key=results.get
    )

    best_accuracy = results[
        best_model
    ]

    print()

    print(
        "===== BEST MODEL ====="
    )

    print()

    print(
        f"Best Model: {best_model}"
    )

    print(
        f"Test Accuracy: {best_accuracy:.2f}%"
    )

    print()

    print(
        "===== COMPARISON COMPLETE ====="
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()

