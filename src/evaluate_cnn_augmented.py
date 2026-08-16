
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from cnn_model import MNISTCNN
from dataset import get_dataloaders


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_mnist_cnn_augmented.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load Data
    # ========================================================

    (
        train_loader,
        validation_loader,
        test_loader
    ) = get_dataloaders()

    print()

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

    # ========================================================
    # Create Model
    # ========================================================

    model = MNISTCNN().to(device)

    # ========================================================
    # Load Best Augmented CNN
    # ========================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print()
    print(
        "Augmented CNN model loaded successfully."
    )

    # ========================================================
    # Loss Function
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Evaluation
    # ========================================================

    test_loss = 0.0

    correct = 0

    total = 0

    all_predictions = []

    all_labels = []

    print()
    print(
        "===== EVALUATING CNN WITH DATA AUGMENTATION ====="
    )

    print()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )

            test_loss += (
                loss.item() * images.size(0)
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # ========================================================
    # Final Metrics
    # ========================================================

    test_loss = test_loss / total

    test_accuracy = (
        100.0 * correct / total
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        "===== AUGMENTED CNN TEST RESULTS ====="
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    print(
        f"Correct predictions: {correct}"
    )

    print(
        f"Total predictions: {total}"
    )

    # ========================================================
    # Classification Report
    # ========================================================

    print()

    print(
        "===== AUGMENTED CNN CLASSIFICATION REPORT ====="
    )

    report = classification_report(
        all_labels,
        all_predictions,
        digits=4
    )

    print(report)

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print(
        "===== CONFUSION MATRIX ====="
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(matrix)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()

