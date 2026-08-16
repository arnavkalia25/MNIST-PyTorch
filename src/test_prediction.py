import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from model import MNISTModel


MODEL_PATH = "models/mnist_mlp.pth"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    # Load dataset
    _, test_loader = get_dataloaders()

    # Load model
    model = MNISTModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    # Get one batch
    images, labels = next(iter(test_loader))

    # Select first image
    image = images[0].unsqueeze(0)
    true_label = labels[0].item()

    image = image.to(device)

    with torch.no_grad():

        logits = model(image)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_digit = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_digit
        ].item()

    print("===== MNIST TEST PREDICTION =====")

    print(
        "Actual digit:",
        true_label
    )

    print(
        "Predicted digit:",
        predicted_digit
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print("\n===== PROBABILITIES =====")

    for digit, probability in enumerate(
        probabilities[0]
    ):

        print(
            f"Digit {digit}: "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":
    main()