
import os

import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model_v2 import MNISTCNNv2
from dataset import get_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 10

LEARNING_RATE = 0.001

MODEL_PATH = "models/best_mnist_cnn_v2.pth"

HISTORY_PATH = "models/cnn_v2_history.pth"


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
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total * 100
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
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total * 100
    )

    return (
        epoch_loss,
        epoch_accuracy
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

    print()
    print(
        "===== CNN V2 MODEL ====="
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
        lr=LEARNING_RATE
    )

    # ========================================================
    # History
    # ========================================================

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "validation_loss": [],

        "validation_accuracy": []
    }

    best_validation_accuracy = 0.0

    # ========================================================
    # Training
    # ========================================================

    print()
    print(
        "===== STARTING CNN V2 TRAINING ====="
    )

    for epoch in range(
        EPOCHS
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion
            )
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "train_accuracy"
        ].append(train_accuracy)

        history[
            "validation_loss"
        ].append(validation_loss)

        history[
            "validation_accuracy"
        ].append(validation_accuracy)

        # ----------------------------------------------------
        # Print epoch
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Learning Rate: "
            f"{LEARNING_RATE:.6f}"
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

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            os.makedirs(
                "models",
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "✓ Best CNN V2 model saved."
            )

    # ========================================================
    # Save history
    # ========================================================

    torch.save(
        history,
        HISTORY_PATH
    )

    # ========================================================
    # Final information
    # ========================================================

    print()
    print(
        "===== CNN V2 TRAINING COMPLETE ====="
    )

    print()

    print(
        "Best model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Training history saved to:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print(
        f"Final training accuracy: "
        f"{history['train_accuracy'][-1]:.2f}%"
    )

    print(
        f"Final validation accuracy: "
        f"{history['validation_accuracy'][-1]:.2f}%"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

