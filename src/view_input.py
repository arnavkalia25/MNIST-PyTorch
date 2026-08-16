import matplotlib.pyplot as plt

from predict import preprocess_image


IMAGE_PATH = "test_images/4.png"


def main():

    image = preprocess_image(IMAGE_PATH)

    print("Processed image shape:", image.shape)

    plt.figure(figsize=(5, 5))

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title("Image Given to Model")

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()