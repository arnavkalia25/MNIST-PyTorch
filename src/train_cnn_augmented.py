
import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import MNISTCNN
from dataset import get_dataloaders


# ============================================================
# Configuration
# ============================================================

EPOCHS = 10

LEARNING_RATE = 0.001

MODEL_PATH = "models/best_mnist_cnn_augmented.pth"

HISTORY_PATH = "models/cnn_augmented_history.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Training Function
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

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = (
        100.0 * correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation Function
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

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = (
        100.0 * correct / total
    )

    return epoch_loss, epoch_accuracy


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
    # Create CNN
    # ========================================================

    model = MNISTCNN().to(device)

    print()
    print("===== CNN WITH DATA AUGMENTATION =====")
    print()

    print(model)

    # ========================================================
    # Loss Function
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
    # Training History
    # ========================================================

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "validation_loss": [],

        "validation_accuracy": []

    }

    # ========================================================
    # Best Validation Accuracy
    # ========================================================

    best_validation_accuracy = 0.0

    # ========================================================
    # Start Training
    # ========================================================

    print()
    print("===== STARTING AUGMENTED CNN TRAINING =====")
    print()

    for epoch in range(EPOCHS):

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Learning Rate: {LEARNING_RATE:.6f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "✓ Best augmented CNN model saved."
            )

        print()

    # ========================================================
    # Save Training History
    # ========================================================

    torch.save(
        history,
        HISTORY_PATH
    )

    # ========================================================
    # Training Complete
    # ========================================================

    print(
        "===== AUGMENTED CNN TRAINING COMPLETE ====="
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

    print()

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
