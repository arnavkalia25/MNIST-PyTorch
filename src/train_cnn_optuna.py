import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import MNISTCNN


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"

BATCH_SIZE = 64

LEARNING_RATE = 0.0010676307787900413

DROPOUT = 0.25452660518957637

EPOCHS = 10

MODEL_PATH = "models/best_mnist_cnn_optuna.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Transform
# ============================================================

transform = transforms.ToTensor()


# ============================================================
# Dataset
# ============================================================

full_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# Use the same 80/20 split as the previous experiments
# ============================================================

train_size = int(
    0.8 * len(full_train_dataset)
)

validation_size = (
    len(full_train_dataset) - train_size
)

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = torch.utils.data.random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=generator
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print()
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# Model
# ============================================================

model = MNISTCNN(
    dropout=DROPOUT
).to(device)


print()
print("============================================================")
print("FINAL CNN WITH OPTUNA HYPERPARAMETERS")
print("============================================================")

print(model)


# ============================================================
# Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training Function
# ============================================================

def train_one_epoch():

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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation Function
# ============================================================

def validate():

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

    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    )

    return validation_loss, validation_accuracy


# ============================================================
# Training
# ============================================================

best_validation_accuracy = 0.0

best_epoch = 0

print()
print("============================================================")
print("STARTING FINAL CNN TRAINING")
print("============================================================")


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = validate()

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    print(
        f"Learning Rate: {LEARNING_RATE:.6f}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best Optuna-tuned CNN model saved."
        )


# ============================================================
# Training Complete
# ============================================================

print()
print("============================================================")
print("FINAL TRAINING COMPLETE")
print("============================================================")

print()
print("Best model saved to:")

print(
    MODEL_PATH
)

print()
print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)

print(
    f"Best epoch: {best_epoch}"
)

print()
print("Optuna hyperparameters used:")

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Dropout: {DROPOUT}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    "Optimizer: Adam"
)