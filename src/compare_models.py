
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results"


# ============================================================
# MODEL RESULTS
# ============================================================

MLP_ACCURACY = 96.67
CNN_ACCURACY = 99.14

MLP_LOSS = 0.1076
CNN_LOSS = 0.0246


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

accuracy_improvement = (
    CNN_ACCURACY - MLP_ACCURACY
)

relative_improvement = (
    accuracy_improvement
    / MLP_ACCURACY
    * 100
)

error_mlp = 100 - MLP_ACCURACY
error_cnn = 100 - CNN_ACCURACY

error_reduction = (
    (error_mlp - error_cnn)
    / error_mlp
    * 100
)


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison():

    print()
    print("=" * 60)
    print("MNIST MODEL COMPARISON")
    print("=" * 60)

    print()

    print(
        f"{'Metric':<25}"
        f"{'MLP':>15}"
        f"{'CNN':>15}"
    )

    print("-" * 60)

    print(
        f"{'Test Accuracy':<25}"
        f"{MLP_ACCURACY:>14.2f}%"
        f"{CNN_ACCURACY:>14.2f}%"
    )

    print(
        f"{'Test Loss':<25}"
        f"{MLP_LOSS:>15.4f}"
        f"{CNN_LOSS:>15.4f}"
    )

    print(
        f"{'Error Rate':<25}"
        f"{error_mlp:>14.2f}%"
        f"{error_cnn:>14.2f}%"
    )

    print("-" * 60)

    print()

    print("===== IMPROVEMENT =====")

    print(
        f"Accuracy improvement: "
        f"+{accuracy_improvement:.2f} percentage points"
    )

    print(
        f"Relative accuracy improvement: "
        f"+{relative_improvement:.2f}%"
    )

    print(
        f"Error reduction: "
        f"{error_reduction:.2f}%"
    )

    print()

    print("===== FINAL MODEL =====")

    if CNN_ACCURACY > MLP_ACCURACY:

        print(
            "CNN is the recommended final model."
        )

        print(
            f"CNN accuracy: {CNN_ACCURACY:.2f}%"
        )

    else:

        print(
            "MLP is the recommended final model."
        )

        print(
            f"MLP accuracy: {MLP_ACCURACY:.2f}%"
        )

    print()


# ============================================================
# ACCURACY COMPARISON
# ============================================================

def plot_accuracy_comparison():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    models = [
        "MLP",
        "CNN"
    ]

    accuracies = [
        MLP_ACCURACY,
        CNN_ACCURACY
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        models,
        accuracies
    )

    plt.ylim(
        90,
        100
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.title(
        "MLP vs CNN Test Accuracy"
    )

    for bar, accuracy in zip(
        bars,
        accuracies
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.1,
            f"{accuracy:.2f}%",
            ha="center",
            fontsize=12
        )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "mlp_vs_cnn_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()


# ============================================================
# LOSS COMPARISON
# ============================================================

def plot_loss_comparison():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    models = [
        "MLP",
        "CNN"
    ]

    losses = [
        MLP_LOSS,
        CNN_LOSS
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        models,
        losses
    )

    plt.ylabel(
        "Test Loss"
    )

    plt.title(
        "MLP vs CNN Test Loss"
    )

    for bar, loss in zip(
        bars,
        losses
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.002,
            f"{loss:.4f}",
            ha="center",
            fontsize=12
        )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "mlp_vs_cnn_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()


# ============================================================
# ERROR RATE COMPARISON
# ============================================================

def plot_error_comparison():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    models = [
        "MLP",
        "CNN"
    ]

    errors = [
        error_mlp,
        error_cnn
    ]

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        models,
        errors
    )

    plt.ylabel(
        "Error Rate (%)"
    )

    plt.title(
        "MLP vs CNN Error Rate"
    )

    for bar, error in zip(
        bars,
        errors
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.02,
            f"{error:.2f}%",
            ha="center",
            fontsize=12
        )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "mlp_vs_cnn_error.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved: {output_path}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print_comparison()

    plot_accuracy_comparison()

    plot_loss_comparison()

    plot_error_comparison()

    print()
    print(
        "===== MODEL COMPARISON COMPLETE ====="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

