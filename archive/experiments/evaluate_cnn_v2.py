
import torch
import torch.nn as nn

from cnn_model_v2 import MNISTCNNv2
from dataset import get_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_cnn_v2.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    test_loss = (
        total_loss / total
    )

    test_accuracy = (
        correct / total * 100
    )

    return (
        test_loss,
        test_accuracy,
        correct,
        total
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Data
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
    # Model
    # ========================================================

    model = MNISTCNNv2()

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
        "CNN V2 model loaded successfully."
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Evaluate
    # ========================================================

    (
        test_loss,
        test_accuracy,
        correct,
        total
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print(
        "===== CNN V2 TEST RESULTS ====="
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

    print()
    print(
        "===== BASELINE COMPARISON ====="
    )

    print(
        "CNN V1 Accuracy: 99.14%"
    )

    print(
        f"CNN V2 Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    difference = (
        test_accuracy - 99.14
    )

    print(
        f"Difference: "
        f"{difference:+.2f} percentage points"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

