import optuna


# ============================================================
# Configuration
# ============================================================

STUDY_NAME = "mnist_cnn_optimization"

STORAGE = "sqlite:///models/optuna_study.db"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load study
    # --------------------------------------------------------

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE
    )

    # --------------------------------------------------------
    # Get completed trials
    # --------------------------------------------------------

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    pruned_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.PRUNED
    ]

    failed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.FAIL
    ]

    # --------------------------------------------------------
    # Number of trials
    # --------------------------------------------------------

    print(
        f"Total trials: {len(study.trials)}"
    )

    print(
        f"Completed trials: {len(completed_trials)}"
    )

    print(
        f"Pruned trials: {len(pruned_trials)}"
    )

    print(
        f"Failed trials: {len(failed_trials)}"
    )

    print()

    # --------------------------------------------------------
    # No completed trials
    # --------------------------------------------------------

    if not completed_trials:

        print(
            "No completed trials available."
        )

        return

    # --------------------------------------------------------
    # Find best completed trial manually
    # --------------------------------------------------------

    best_trial = max(
        completed_trials,
        key=lambda trial: trial.value
    )

    # --------------------------------------------------------
    # Best trial
    # --------------------------------------------------------

    print("=" * 70)
    print("BEST TRIAL")
    print("=" * 70)

    print()

    print(
        f"Best trial number: {best_trial.number}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_trial.value * 100:.2f}%"
    )

    print()

    print("Best hyperparameters:")

    for parameter, value in best_trial.params.items():

        print(
            f"  {parameter}: {value}"
        )

    print()

    # --------------------------------------------------------
    # All completed trials
    # --------------------------------------------------------

    print("=" * 70)
    print("COMPLETED TRIALS")
    print("=" * 70)

    print()

    for trial in completed_trials:

        print(
            f"Trial {trial.number}: "
            f"{trial.value * 100:.2f}%"
        )

        for parameter, value in trial.params.items():

            print(
                f"    {parameter}: {value}"
            )

        print()

    # --------------------------------------------------------
    # Pruned trials
    # --------------------------------------------------------

    if pruned_trials:

        print("=" * 70)
        print("PRUNED TRIALS")
        print("=" * 70)

        print()

        for trial in pruned_trials:

            print(
                f"Trial {trial.number}: PRUNED"
            )

        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("OPTUNA ANALYSIS COMPLETE")
    print("=" * 70)

    print()

    print(
        "Best validation accuracy: "
        f"{best_trial.value * 100:.2f}%"
    )

    print()

    print("Best configuration:")

    for parameter, value in best_trial.params.items():

        print(
            f"  {parameter}: {value}"
        )

    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()