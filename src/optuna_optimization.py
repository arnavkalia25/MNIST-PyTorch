import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import get_dataloaders
from cnn_model_optuna import MNISTCNNOptuna


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

STUDY_NAME = "mnist_cnn_optimization_v2"

STORAGE = "sqlite:///models/optuna_study_v2.db"

N_TRIALS = 10

EPOCHS = 3


# ============================================================
# Objective Function
# ============================================================

def objective(trial):

    # ========================================================
    # Hyperparameters
    # ========================================================

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.1,
        0.5
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "SGD"]
    )

    conv1_channels = trial.suggest_categorical(
        "conv1_channels",
        [16, 32, 48]
    )

    conv2_channels = trial.suggest_categorical(
        "conv2_channels",
        [32, 64, 96]
    )

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        [64, 128, 256]
    )

    # ========================================================
    # Load datasets
    # ========================================================

    train_loader_original, validation_loader, _ = (
        get_dataloaders()
    )

    # ========================================================
    # Rebuild train loader with Optuna batch size
    # ========================================================

    train_dataset = train_loader_original.dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # ========================================================
    # Model
    # ========================================================

    model = MNISTCNNOptuna(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(DEVICE)

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Optimizer
    # ========================================================

    if optimizer_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    # ========================================================
    # Training
    # ========================================================

    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
                )

                validation_total += labels.size(0)

                validation_correct += (
                    predictions == labels
                ).sum().item()

        validation_accuracy = (
            validation_correct /
            validation_total
        )

        best_validation_accuracy = max(
            best_validation_accuracy,
            validation_accuracy
        )

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        # ====================================================
        # Report to Optuna
        # ====================================================

        trial.report(
            validation_accuracy,
            epoch
        )

        # ====================================================
        # Pruning
        # ====================================================

        if trial.should_prune():

            raise optuna.exceptions.TrialPruned()

    print()

    print(
        f"Trial {trial.number} complete | "
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print()

    return best_validation_accuracy


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "OPTUNA CNN HYPERPARAMETER OPTIMIZATION V2"
    )

    print("=" * 70)

    print()

    print(
        f"Using device: {DEVICE}"
    )

    print(
        f"Number of trials: {N_TRIALS}"
    )

    print(
        f"Epochs per trial: {EPOCHS}"
    )

    print()

    # ========================================================
    # Create study
    # ========================================================

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1
        )
    )

    # ========================================================
    # Optimization
    # ========================================================

    study.optimize(
        objective,
        n_trials=N_TRIALS
    )

    # ========================================================
    # Results
    # ========================================================

    print()

    print("=" * 70)

    print(
        "OPTIMIZATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Total trials: {len(study.trials)}"
    )

    print()

    print(
        f"Best validation accuracy: "
        f"{study.best_value * 100:.2f}%"
    )

    print()

    print(
        "Best hyperparameters:"
    )

    for parameter, value in study.best_params.items():

        print(
            f"  {parameter}: {value}"
        )

    print()

    print(
        "Study saved to:"
    )

    print(
        "models/optuna_study_v2.db"
    )

    print()

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()