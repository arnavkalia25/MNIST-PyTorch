import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import MNISTModel


MODEL_PATH = "models/mnist_mlp.pth"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    _, test_loader = get_dataloaders()

    model = MNISTModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    images, labels = next(iter(test_loader))

    image = images[0]

    with torch.no_grad():

        output = model(
            image.unsqueeze(0).to(device)
        )

        prediction = output.argmax(
            dim=1
        ).item()

    actual = labels[0].item()

    plt.figure(figsize=(5, 5))

    plt.imshow(
        image.squeeze(0),
        cmap="gray"
    )

    plt.title(
        f"Actual: {actual} | "
        f"Predicted: {prediction}"
    )

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()