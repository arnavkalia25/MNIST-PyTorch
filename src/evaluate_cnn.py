
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from cnn_model import MNISTCNN
from dataset import get_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_cnn.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Device
    # ========================================================

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Check model
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nCNN model not found: "
            f"{MODEL_PATH}\n\n"
            "Please train the CNN first using:\n"
            "python src\\train_cnn.py"
        )

    # ========================================================
    # Load data
    # ========================================================

    (
        train_loader,
        validation_loader,
        test_loader
    ) = get_dataloaders()

    print()

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # ========================================================
    # Create CNN
    # ========================================================

    model = MNISTCNN()

    model = model.to(device)

    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    print()

    print(
        "CNN model loaded successfully."
    )

    # ========================================================
    # Evaluation mode
    # ========================================================

    model.eval()

    # ========================================================
    # Loss function
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Statistics
    # ========================================================

    total_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []

    # ========================================================
    # Test loop
    # ========================================================

    print()

    print(
        "===== EVALUATING CNN ====="
    )

    with torch.no_grad():

        for images, labels in test_loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    # ========================================================
    # Calculate metrics
    # ========================================================

    test_loss = (
        total_loss /
        len(test_loader)
    )

    test_accuracy = (
        100.0 *
        correct /
        total
    )

    # ========================================================
    # Test results
    # ========================================================

    print()

    print(
        "===== CNN TEST RESULTS ====="
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
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

    # ========================================================
    # Classification report
    # ========================================================

    print()

    print(
        "===== CNN CLASSIFICATION REPORT ====="
    )

    report = classification_report(
        all_labels,
        all_predictions,
        digits=4
    )

    print(
        report
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print()

    print(
        "===== CONFUSION MATRIX ====="
    )

    print(
        cm
    )

    # ========================================================
    # Display confusion matrix
    # ========================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    display.plot()

    plt.title(
        "MNIST CNN Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

