
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MNISTCNN


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_cnn.pth"

DATA_DIR = "data"

RESULTS_DIR = "results"

BATCH_SIZE = 64

NUM_CLASSES = 10

NUM_IMAGES_TO_SHOW = 25


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

def get_test_loader():

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

    return (
        test_dataset,
        test_loader
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = MNISTCNN()

    model = model.to(
        device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

def collect_predictions(
    model,
    test_loader
):

    all_images = []

    all_labels = []

    all_predictions = []

    all_confidences = []

    total_correct = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, predictions = torch.max(
                probabilities,
                dim=1
            )

            total_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total_samples += (
                labels.size(0)
            )

            all_images.append(
                images.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_confidences.append(
                confidence.cpu()
            )

    # ========================================================
    # Combine batches
    # ========================================================

    all_images = torch.cat(
        all_images
    )

    all_labels = torch.cat(
        all_labels
    )

    all_predictions = torch.cat(
        all_predictions
    )

    all_confidences = torch.cat(
        all_confidences
    )

    accuracy = (
        total_correct
        / total_samples
        * 100
    )

    return (
        all_images,
        all_labels,
        all_predictions,
        all_confidences,
        accuracy
    )


# ============================================================
# FIND MISCLASSIFIED IMAGES
# ============================================================

def get_misclassified(
    images,
    labels,
    predictions,
    confidences
):

    incorrect_mask = (
        predictions != labels
    )

    incorrect_indices = (
        torch.where(
            incorrect_mask
        )[0]
    )

    return incorrect_indices


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

def calculate_class_accuracy(
    labels,
    predictions
):

    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    for label, prediction in zip(
        labels.numpy(),
        predictions.numpy()
    ):

        class_total[label] += 1

        if label == prediction:

            class_correct[label] += 1

    class_accuracy = (
        class_correct
        / class_total
        * 100
    )

    return (
        class_correct,
        class_total,
        class_accuracy
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    labels,
    predictions
):

    matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    for actual, predicted in zip(
        labels.numpy(),
        predictions.numpy()
    ):

        matrix[
            actual,
            predicted
        ] += 1

    return matrix


# ============================================================
# SAVE ERROR REPORT
# ============================================================

def save_error_report(
    labels,
    predictions,
    confidences,
    incorrect_indices,
    class_correct,
    class_total,
    class_accuracy,
    confusion_matrix,
    overall_accuracy
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "cnn_error_analysis.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "========================================\n"
        )

        file.write(
            "CNN ERROR ANALYSIS REPORT\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            f"Overall Test Accuracy: "
            f"{overall_accuracy:.2f}%\n"
        )

        file.write(
            f"Total Test Images: "
            f"{len(labels)}\n"
        )

        file.write(
            f"Correct Predictions: "
            f"{len(labels) - len(incorrect_indices)}\n"
        )

        file.write(
            f"Incorrect Predictions: "
            f"{len(incorrect_indices)}\n\n"
        )

        # ====================================================
        # Per-class accuracy
        # ====================================================

        file.write(
            "========================================\n"
        )

        file.write(
            "PER-CLASS ACCURACY\n"
        )

        file.write(
            "========================================\n\n"
        )

        for digit in range(NUM_CLASSES):

            file.write(
                f"Digit {digit}: "
                f"{class_accuracy[digit]:.2f}% "
                f"("
                f"{class_correct[digit]}/"
                f"{class_total[digit]}"
                f")\n"
            )

        file.write("\n")

        # ====================================================
        # Most common errors
        # ====================================================

        file.write(
            "========================================\n"
        )

        file.write(
            "MOST COMMON CONFUSIONS\n"
        )

        file.write(
            "========================================\n\n"
        )

        confusion_pairs = []

        for actual in range(NUM_CLASSES):

            for predicted in range(NUM_CLASSES):

                if actual == predicted:

                    continue

                count = confusion_matrix[
                    actual,
                    predicted
                ]

                confusion_pairs.append(
                    (
                        count,
                        actual,
                        predicted
                    )
                )

        confusion_pairs.sort(
            reverse=True
        )

        for (
            count,
            actual,
            predicted
        ) in confusion_pairs[:15]:

            if count > 0:

                file.write(
                    f"{actual} → {predicted}: "
                    f"{count} images\n"
                )

        file.write("\n")

        # ====================================================
        # Highest confidence errors
        # ====================================================

        file.write(
            "========================================\n"
        )

        file.write(
            "HIGH-CONFIDENCE WRONG PREDICTIONS\n"
        )

        file.write(
            "========================================\n\n"
        )

        wrong_confidences = []

        for index in incorrect_indices:

            index_value = index.item()

            wrong_confidences.append(
                (
                    confidences[
                        index_value
                    ].item(),
                    index_value,
                    labels[
                        index_value
                    ].item(),
                    predictions[
                        index_value
                    ].item()
                )
            )

        wrong_confidences.sort(
            reverse=True
        )

        for (
            confidence,
            index,
            actual,
            predicted
        ) in wrong_confidences[:25]:

            file.write(
                f"Index {index}: "
                f"Actual={actual}, "
                f"Predicted={predicted}, "
                f"Confidence="
                f"{confidence * 100:.2f}%\n"
            )

    print(
        f"\nError report saved to:\n"
        f"{report_path}"
    )


# ============================================================
# VISUALIZE MISCLASSIFIED IMAGES
# ============================================================

def visualize_misclassified(
    images,
    labels,
    predictions,
    confidences,
    incorrect_indices
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    number_to_show = min(
        NUM_IMAGES_TO_SHOW,
        len(incorrect_indices)
    )

    # ========================================================
    # Select highest-confidence wrong predictions
    # ========================================================

    selected = sorted(
        incorrect_indices.tolist(),
        key=lambda index:
        confidences[index].item(),
        reverse=True
    )[:number_to_show]

    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(12, 12)
    )

    axes = axes.flatten()

    for position, index in enumerate(
        selected
    ):

        image = images[
            index
        ].squeeze()

        actual = labels[
            index
        ].item()

        predicted = predictions[
            index
        ].item()

        confidence = (
            confidences[
                index
            ].item()
            * 100
        )

        axes[position].imshow(
            image,
            cmap="gray"
        )

        axes[position].set_title(
            (
                f"Actual: {actual}\n"
                f"Pred: {predicted}\n"
                f"Conf: {confidence:.1f}%"
            )
        )

        axes[position].axis(
            "off"
        )

    # ========================================================
    # Hide unused axes
    # ========================================================

    for position in range(
        len(selected),
        len(axes)
    ):

        axes[position].axis(
            "off"
        )

    fig.suptitle(
        "CNN Misclassified Images",
        fontsize=18
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "cnn_misclassified_images.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Misclassified image visualization saved to:\n"
        f"{output_path}"
    )

    plt.show()


# ============================================================
# VISUALIZE PER-CLASS ACCURACY
# ============================================================

def visualize_class_accuracy(
    class_accuracy
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    digits = np.arange(
        NUM_CLASSES
    )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        digits,
        class_accuracy
    )

    plt.ylim(
        90,
        100
    )

    plt.xlabel(
        "Digit"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "CNN Accuracy by Digit"
    )

    plt.xticks(
        digits
    )

    # ========================================================
    # Add values above bars
    # ========================================================

    for bar, accuracy in zip(
        bars,
        class_accuracy
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.05,
            f"{accuracy:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "cnn_class_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Class accuracy chart saved to:\n"
        f"{output_path}"
    )

    plt.show()


# ============================================================
# VISUALIZE CONFUSION MATRIX
# ============================================================

def visualize_confusion_matrix(
    confusion_matrix
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.imshow(
        confusion_matrix,
        cmap="Blues"
    )

    plt.title(
        "CNN Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Digit"
    )

    plt.ylabel(
        "Actual Digit"
    )

    plt.xticks(
        range(NUM_CLASSES)
    )

    plt.yticks(
        range(NUM_CLASSES)
    )

    # ========================================================
    # Add values
    # ========================================================

    for actual in range(
        NUM_CLASSES
    ):

        for predicted in range(
            NUM_CLASSES
        ):

            plt.text(
                predicted,
                actual,
                confusion_matrix[
                    actual,
                    predicted
                ],
                ha="center",
                va="center",
                fontsize=8
            )

    plt.colorbar()

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "cnn_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Confusion matrix saved to:\n"
        f"{output_path}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Load data
    # ========================================================

    (
        test_dataset,
        test_loader
    ) = get_test_loader()

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model()

    print(
        "CNN model loaded successfully."
    )

    # ========================================================
    # Collect predictions
    # ========================================================

    print()
    print(
        "===== RUNNING ERROR ANALYSIS ====="
    )

    (
        images,
        labels,
        predictions,
        confidences,
        accuracy
    ) = collect_predictions(
        model,
        test_loader
    )

    # ========================================================
    # Find errors
    # ========================================================

    incorrect_indices = (
        get_misclassified(
            images,
            labels,
            predictions,
            confidences
        )
    )

    # ========================================================
    # Class accuracy
    # ========================================================

    (
        class_correct,
        class_total,
        class_accuracy
    ) = calculate_class_accuracy(
        labels,
        predictions
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    confusion_matrix = (
        calculate_confusion_matrix(
            labels,
            predictions
        )
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()

    print(
        "===== ERROR ANALYSIS RESULTS ====="
    )

    print(
        f"Overall Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Correct Predictions: "
        f"{len(labels) - len(incorrect_indices)}"
    )

    print(
        f"Incorrect Predictions: "
        f"{len(incorrect_indices)}"
    )

    print()

    # ========================================================
    # Per-class accuracy
    # ========================================================

    print(
        "===== PER-CLASS ACCURACY ====="
    )

    for digit in range(
        NUM_CLASSES
    ):

        print(
            f"Digit {digit}: "
            f"{class_accuracy[digit]:.2f}% "
            f"("
            f"{class_correct[digit]}/"
            f"{class_total[digit]}"
            f")"
        )

    # ========================================================
    # Save report
    # ========================================================

    save_error_report(
        labels,
        predictions,
        confidences,
        incorrect_indices,
        class_correct,
        class_total,
        class_accuracy,
        confusion_matrix,
        accuracy
    )

    # ========================================================
    # Visualizations
    # ========================================================

    visualize_misclassified(
        images,
        labels,
        predictions,
        confidences,
        incorrect_indices
    )

    visualize_class_accuracy(
        class_accuracy
    )

    visualize_confusion_matrix(
        confusion_matrix
    )

    print()

    print(
        "===== ERROR ANALYSIS COMPLETE ====="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

