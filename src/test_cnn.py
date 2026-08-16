
import torch

from cnn_model import MNISTCNN


def main():

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Create model
    # ========================================================

    model = MNISTCNN()

    model = model.to(device)

    print()
    print(
        "===== CNN ARCHITECTURE ====="
    )

    print(model)

    # ========================================================
    # Create dummy MNIST image
    # ========================================================

    x = torch.randn(
        1,
        1,
        28,
        28
    ).to(device)

    # ========================================================
    # Forward pass
    # ========================================================

    with torch.no_grad():

        output = model(x)

    # ========================================================
    # Display shapes
    # ========================================================

    print()

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Output shape: {output.shape}"
    )

    print()

    print(
        "CNN forward pass successful!"
    )


if __name__ == "__main__":

    main()

