import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from cnn_model import MNISTCNN


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"

BATCH_SIZE = 64

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
# Same train/validation split
# ============================================================

train_size = int(
    0.8 * len(full_train_dataset)
)

validation_size = (
    len(full_train_dataset) - train_size
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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
    dropout=0.25452660518957637
).to(device)


# ============================================================
# Load trained model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print()
print("Optuna-tuned CNN model loaded successfully.")


# ============================================================
# Loss Function
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Evaluation
# ============================================================

total_loss = 0.0

correct = 0

total = 0

all_predictions = []

all_labels = []


print()
print("============================================================")
print("EVALUATING OPTUNA-TUNED CNN")
print("============================================================")


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

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# Calculate metrics
# ============================================================

test_loss = total_loss / total

test_accuracy = correct / total


# ============================================================
# Test Results
# ============================================================

print()
print("============================================================")
print("OPTUNA CNN TEST RESULTS")
print("============================================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Total predictions: {total}"
)


# ============================================================
# Classification Report
# ============================================================

print()
print("============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")

report = classification_report(
    all_labels,
    all_predictions,
    digits=4
)

print(report)


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("============================================================")
print("CONFUSION MATRIX")
print("============================================================")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)


# ============================================================
# Per-class accuracy
# ============================================================

print()
print("============================================================")
print("PER-CLASS ACCURACY")
print("============================================================")

for digit in range(10):

    total_digit = cm[digit].sum()

    correct_digit = cm[digit][digit]

    accuracy = (
        correct_digit / total_digit
    ) * 100

    print(
        f"Digit {digit}: "
        f"{accuracy:.2f}%"
    )


# ============================================================
# Complete
# ============================================================

print()
print("============================================================")
print("EVALUATION COMPLETE")
print("============================================================")