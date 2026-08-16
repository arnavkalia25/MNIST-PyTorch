
import os

import torch

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_PATH = "models/cnn_history.pth"

OUTPUT_DIR = "results"


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Check history
    # ========================================================

    if not os.path.exists(
        HISTORY_PATH
    ):

        raise FileNotFoundError(
            f"\nTraining history not found:\n"
            f"{HISTORY_PATH}\n\n"
            "Run:\n"
            "python src\\train_cnn.py"
        )

    # ========================================================
    # Create results directory
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # Load history
    # ========================================================

    history = torch.load(
        HISTORY_PATH,
        map_location="cpu"
    )

    train_losses = history[
        "train_losses"
    ]

    validation_losses = history[
        "validation_losses"
    ]

    train_accuracies = history[
        "train_accuracies"
    ]

    validation_accuracies = history[
        "validation_accuracies"
    ]

    # ========================================================
    # Epoch numbers
    # ========================================================

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # ========================================================
    # Loss curve
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    loss_path = os.path.join(
        OUTPUT_DIR,
        "cnn_loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=300
    )

    print(
        f"Saved: {loss_path}"
    )

    plt.show()

    # ========================================================
    # Accuracy curve
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracies,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        validation_accuracies,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "CNN Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    accuracy_path = os.path.join(
        OUTPUT_DIR,
        "cnn_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300
    )

    print(
        f"Saved: {accuracy_path}"
    )

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

