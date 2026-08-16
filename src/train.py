import csv

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import MNISTModel


# =========================
# Configuration
# =========================

LEARNING_RATE = 0.001

EPOCHS = 30

PATIENCE = 5

MIN_DELTA = 0.0001

MODEL_PATH = "models/best_mnist_mlp.pth"

HISTORY_PATH = "training_history.csv"


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Training Function
# =========================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Loss
        running_loss += loss.item()

        # Predictions
        predictions = outputs.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100.0 * correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# =========================
# Validation Function
# =========================

def validate(
    model,
    validation_loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    validation_loss = (
        running_loss /
        len(validation_loader)
    )

    validation_accuracy = (
        100.0 * correct / total
    )

    return (
        validation_loss,
        validation_accuracy
    )


# =========================
# Save Training History
# =========================

def save_history(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies,
    learning_rates
):

    with open(
        HISTORY_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "validation_loss",
            "train_accuracy",
            "validation_accuracy",
            "learning_rate"
        ])

        for i in range(
            len(train_losses)
        ):

            writer.writerow([
                i + 1,
                train_losses[i],
                validation_losses[i],
                train_accuracies[i],
                validation_accuracies[i],
                learning_rates[i]
            ])


# =========================
# Plot Training Curves
# =========================

def plot_training_history(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies
):

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # -------------------------
    # Loss
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "training_loss.png"
    )

    plt.show()

    # -------------------------
    # Accuracy
    # -------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "training_accuracy.png"
    )

    plt.show()


# =========================
# Main
# =========================

def main():

    print(
        "Using device:",
        device
    )

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

    print("\nModel:")

    print(model)

    # -------------------------
    # Loss function
    # -------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # -------------------------
    # Training history
    # -------------------------

    train_losses = []

    validation_losses = []

    train_accuracies = []

    validation_accuracies = []

    learning_rates = []

    # -------------------------
    # Best model
    # -------------------------

    best_validation_loss = float(
        "inf"
    )

    # -------------------------
    # Early stopping
    # -------------------------

    patience_counter = 0

    # -------------------------
    # Training
    # -------------------------

    print(
        "\n===== TRAINING ====="
    )

    for epoch in range(EPOCHS):

        # =====================
        # Train
        # =====================

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # =====================
        # Validate
        # =====================

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        # =====================
        # Scheduler
        # =====================

        scheduler.step(
            validation_loss
        )

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # =====================
        # Store history
        # =====================

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            validation_accuracy
        )

        learning_rates.append(
            current_lr
        )

        # =====================
        # Print epoch
        # =====================

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {validation_loss:.4f} "
            f"Val Acc: {validation_accuracy:.2f}% "
            f"LR: {current_lr:.6f}"
        )

        # =====================
        # Check improvement
        # =====================

        if (
            validation_loss
            <
            best_validation_loss - MIN_DELTA
        ):

            best_validation_loss = (
                validation_loss
            )

            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "  → Best model saved!"
            )

        else:

            patience_counter += 1

            print(
                f"  → No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # =====================
        # Early stopping
        # =====================

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Training stopped at "
                f"epoch {epoch + 1}."
            )

            break

    # -------------------------
    # Save history
    # -------------------------

    save_history(
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        learning_rates
    )

    # -------------------------
    # Plot graphs
    # -------------------------

    plot_training_history(
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies
    )

    print(
        "\n===== TRAINING COMPLETE ====="
    )

    print(
        "Best model:",
        MODEL_PATH
    )

    print(
        "Training history:",
        HISTORY_PATH
    )


if __name__ == "__main__":
    main()