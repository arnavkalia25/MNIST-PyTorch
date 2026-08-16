
import os

import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import MNISTCNN
from dataset import get_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_cnn.pth"

HISTORY_PATH = "models/cnn_history.pth"

EPOCHS = 10

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

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

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

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
        100.0 *
        correct /
        total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

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
        100.0 *
        correct /
        total
    )

    return (
        validation_loss,
        validation_accuracy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Create models directory
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
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

    print()

    print(
        "===== CNN MODEL ====="
    )

    print(model)

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # ========================================================
    # Best validation loss
    # ========================================================

    best_validation_loss = float(
        "inf"
    )

    # ========================================================
    # Training history
    # ========================================================

    train_losses = []

    train_accuracies = []

    validation_losses = []

    validation_accuracies = []

    learning_rates = []

    # ========================================================
    # Training
    # ========================================================

    print()

    print(
        "===== STARTING CNN TRAINING ====="
    )

    for epoch in range(
        EPOCHS
    ):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        train_losses.append(
            train_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        validation_losses.append(
            validation_loss
        )

        validation_accuracies.append(
            validation_accuracy
        )

        learning_rates.append(
            current_lr
        )

        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "✓ Best CNN model saved."
            )

    # ========================================================
    # Save training history
    # ========================================================

    history = {

        "train_losses": train_losses,

        "train_accuracies": train_accuracies,

        "validation_losses": validation_losses,

        "validation_accuracies": validation_accuracies,

        "learning_rates": learning_rates
    }

    torch.save(
        history,
        HISTORY_PATH
    )

    # ========================================================
    # Training complete
    # ========================================================

    print()

    print(
        "===== TRAINING COMPLETE ====="
    )

    print(
        f"Best model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        f"Training history saved to:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print(
        f"Final training accuracy: "
        f"{train_accuracies[-1]:.2f}%"
    )

    print(
        f"Final validation accuracy: "
        f"{validation_accuracies[-1]:.2f}%"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
