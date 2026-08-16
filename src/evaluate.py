import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import MNISTModel


# =========================
# Configuration
# =========================

MODEL_PATH = "models/best_mnist_mlp.pth"


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Main
# =========================

def main():

    print("Using device:", device)

    # -------------------------
    # Load datasets
    # -------------------------

    (
        train_loader,
        validation_loader,
        test_loader
    ) = get_dataloaders()

    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset)
    )

    print(
        "Test samples:",
        len(test_loader.dataset)
    )

    # -------------------------
    # Create model
    # -------------------------

    model = MNISTModel()

    model = model.to(device)

    print("\nModel architecture:")
    print(model)

    # -------------------------
    # Load checkpoint
    # -------------------------

    print(
        "\nLoading checkpoint:"
    )

    print(
        MODEL_PATH
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # -------------------------
    # Verify checkpoint
    # -------------------------

    print(
        "\nCheckpoint first keys:"
    )

    print(
        list(checkpoint.keys())[:10]
    )

    # -------------------------
    # Load weights
    # -------------------------

    model.load_state_dict(
        checkpoint
    )

    print(
        "\nBest model loaded successfully."
    )

    # -------------------------
    # Evaluation mode
    # -------------------------

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []

    # -------------------------
    # Evaluate
    # -------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    # -------------------------
    # Results
    # -------------------------

    test_loss = (
        total_loss /
        len(test_loader)
    )

    test_accuracy = (
        100.0 *
        correct /
        total
    )

    print(
        "\n===== TEST RESULTS ====="
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Correct predictions: "
        f"{correct}"
    )

    print(
        f"Total predictions: "
        f"{total}"
    )

    # -------------------------
    # Classification report
    # -------------------------

    print(
        "\n===== CLASSIFICATION REPORT ====="
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            digits=4
        )
    )

    # -------------------------
    # Confusion matrix
    # -------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    display.plot()

    plt.title(
        "MNIST Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()