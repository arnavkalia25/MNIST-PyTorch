import torch
from pathlib import Path

from dataset import get_dataloaders
from model import MNISTModel
from cnn_model import MNISTCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLP_PATH = "models/best_mnist_mlp.pth"
CNN_PATH = "models/best_mnist_cnn.pth"
AUGMENTED_CNN_PATH = "models/best_mnist_cnn_augmented.pth"

OUTPUT_PATH = "results/final_model_comparison.txt"


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total

    accuracy = 100.0 * correct / total

    errors = total - correct

    return average_loss, accuracy, errors


# ============================================================
# LOAD MLP
# ============================================================

def load_mlp():

    model = MNISTModel().to(DEVICE)

    checkpoint = torch.load(
        MLP_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# LOAD CNN
# ============================================================

def load_cnn():

    model = MNISTCNN().to(DEVICE)

    checkpoint = torch.load(
        CNN_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# LOAD AUGMENTED CNN
# ============================================================

def load_augmented_cnn():

    model = MNISTCNN().to(DEVICE)

    checkpoint = torch.load(
        AUGMENTED_CNN_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL MNIST MODEL COMPARISON")
    print("=" * 70)

    print()

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    Path("results").mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    _, _, test_loader = get_dataloaders()

    print()

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()

    print("Loading models...")

    mlp = load_mlp()

    print("✓ MLP loaded")

    cnn = load_cnn()

    print("✓ CNN loaded")

    augmented_cnn = load_augmented_cnn()

    print("✓ CNN + Augmentation loaded")

    # --------------------------------------------------------
    # Evaluate MLP
    # --------------------------------------------------------

    print()

    print("Evaluating MLP...")

    mlp_loss, mlp_accuracy, mlp_errors = evaluate_model(
        mlp,
        test_loader
    )

    # --------------------------------------------------------
    # Evaluate CNN
    # --------------------------------------------------------

    print("Evaluating CNN...")

    cnn_loss, cnn_accuracy, cnn_errors = evaluate_model(
        cnn,
        test_loader
    )

    # --------------------------------------------------------
    # Evaluate augmented CNN
    # --------------------------------------------------------

    print(
        "Evaluating CNN + Augmentation..."
    )

    aug_loss, aug_accuracy, aug_errors = evaluate_model(
        augmented_cnn,
        test_loader
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = [
        (
            "MLP",
            mlp_loss,
            mlp_accuracy,
            mlp_errors
        ),
        (
            "CNN",
            cnn_loss,
            cnn_accuracy,
            cnn_errors
        ),
        (
            "CNN + Augmentation",
            aug_loss,
            aug_accuracy,
            aug_errors
        )
    ]

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Model':<25}"
        f"{'Loss':<15}"
        f"{'Accuracy':<15}"
        f"{'Errors':<10}"
    )

    print("-" * 70)

    for name, loss, accuracy, errors in results:

        print(
            f"{name:<25}"
            f"{loss:<15.4f}"
            f"{accuracy:<15.2f}%"
            f"{errors:<10}"
        )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("IMPROVEMENTS")
    print("=" * 70)

    print()

    print(
        f"CNN vs MLP: "
        f"{cnn_accuracy - mlp_accuracy:+.2f} percentage points"
    )

    print(
        f"CNN + Augmentation vs MLP: "
        f"{aug_accuracy - mlp_accuracy:+.2f} percentage points"
    )

    print(
        f"CNN + Augmentation vs CNN: "
        f"{aug_accuracy - cnn_accuracy:+.2f} percentage points"
    )

    # --------------------------------------------------------
    # Find best model
    # --------------------------------------------------------

    best_model = max(
        results,
        key=lambda x: x[2]
    )

    best_name = best_model[0]
    best_loss = best_model[1]
    best_accuracy = best_model[2]
    best_errors = best_model[3]

    print()

    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print()

    print(
        f"Best Model: {best_name}"
    )

    print(
        f"Test Loss: {best_loss:.4f}"
    )

    print(
        f"Test Accuracy: {best_accuracy:.2f}%"
    )

    print(
        f"Incorrect predictions: {best_errors}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FINAL MNIST MODEL COMPARISON\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            f"Test samples: {len(test_loader.dataset)}\n\n"
        )

        file.write(
            f"{'Model':<25}"
            f"{'Loss':<15}"
            f"{'Accuracy':<15}"
            f"{'Errors':<10}\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for name, loss, accuracy, errors in results:

            file.write(
                f"{name:<25}"
                f"{loss:<15.4f}"
                f"{accuracy:<15.2f}%"
                f"{errors:<10}\n"
            )

        file.write("\n")

        file.write(
            f"CNN vs MLP: "
            f"{cnn_accuracy - mlp_accuracy:+.2f} percentage points\n"
        )

        file.write(
            f"CNN + Augmentation vs MLP: "
            f"{aug_accuracy - mlp_accuracy:+.2f} percentage points\n"
        )

        file.write(
            f"CNN + Augmentation vs CNN: "
            f"{aug_accuracy - cnn_accuracy:+.2f} percentage points\n"
        )

        file.write("\n")

        file.write(
            f"Best Model: {best_name}\n"
        )

        file.write(
            f"Best Test Accuracy: "
            f"{best_accuracy:.2f}%\n"
        )

        file.write(
            f"Best Test Loss: "
            f"{best_loss:.4f}\n"
        )

        file.write(
            f"Incorrect Predictions: "
            f"{best_errors}\n"
        )

    print()

    print(
        f"Results saved to: {OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print("FINAL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()