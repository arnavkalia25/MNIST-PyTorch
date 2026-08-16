import matplotlib.pyplot as plt

from dataset import get_dataloaders


def main():

    (
        train_loader,
        validation_loader,
        test_loader
    ) = get_dataloaders()

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

    images, labels = next(
        iter(train_loader)
    )

    image = images[0]
    label = labels[0]

    print("\nImage information:")
    print("Shape:", image.shape)
    print("Label:", label)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(
        f"Digit: {label}"
    )

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()