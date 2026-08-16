
import os

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MNIST_SIZE = 28
DIGIT_SIZE = 20


# ============================================================
# CALCULATE CENTER OF MASS
# ============================================================

def calculate_center_of_mass(image):
    """
    Calculate the weighted center of the digit.

    Returns:
        center_x, center_y
    """

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Total pixel intensity
    total_mass = array.sum()

    # If image is empty
    if total_mass == 0:

        return (
            image.width / 2,
            image.height / 2
        )

    # Coordinate grids
    y_indices, x_indices = np.indices(
        array.shape
    )

    # Weighted coordinates
    center_x = (
        (x_indices * array).sum()
        / total_mass
    )

    center_y = (
        (y_indices * array).sum()
        / total_mass
    )

    return (
        center_x,
        center_y
    )


# ============================================================
# SHIFT IMAGE
# ============================================================

def shift_image(
    image,
    shift_x,
    shift_y
):
    """
    Shift image by the specified amount.
    """

    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (
            1,
            0,
            -shift_x,
            0,
            1,
            -shift_y
        ),
        resample=Image.Resampling.BICUBIC
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    # ========================================================
    # Load image
    # ========================================================

    image = Image.open(
        image_path
    )

    print(
        f"Original image size: "
        f"{image.size}"
    )

    # ========================================================
    # Grayscale
    # ========================================================

    image = image.convert(
        "L"
    )

    # ========================================================
    # Calculate average brightness
    # ========================================================

    pixel_data = list(
        image.getdata()
    )

    mean_pixel = (
        sum(pixel_data)
        / len(pixel_data)
    )

    print(
        f"Average pixel value: "
        f"{mean_pixel:.2f}"
    )

    # ========================================================
    # Invert if necessary
    # ========================================================

    if mean_pixel > 127:

        image = ImageOps.invert(
            image
        )

        print(
            "Image inverted."
        )

    # ========================================================
    # Threshold
    # ========================================================

    threshold = 40

    binary = image.point(
        lambda pixel:
        255 if pixel > threshold
        else 0
    )

    # ========================================================
    # Find bounding box
    # ========================================================

    bbox = binary.getbbox()

    if bbox is None:

        raise ValueError(
            "Could not detect digit."
        )

    print(
        f"Bounding box: {bbox}"
    )

    # ========================================================
    # Crop digit
    # ========================================================

    image = image.crop(
        bbox
    )

    print(
        f"Cropped digit size: "
        f"{image.size}"
    )

    # ========================================================
    # Add padding
    # ========================================================

    width, height = image.size

    padding = int(
        max(width, height)
        * 0.15
    )

    padded_width = (
        width + 2 * padding
    )

    padded_height = (
        height + 2 * padding
    )

    padded_image = Image.new(
        "L",
        (
            padded_width,
            padded_height
        ),
        0
    )

    padded_image.paste(
        image,
        (
            padding,
            padding
        )
    )

    image = padded_image

    # ========================================================
    # Calculate original center of mass
    # ========================================================

    center_x, center_y = (
        calculate_center_of_mass(
            image
        )
    )

    print(
        f"Center of mass before resize: "
        f"({center_x:.2f}, {center_y:.2f})"
    )

    # ========================================================
    # Preserve aspect ratio
    # ========================================================

    width, height = image.size

    if width > height:

        new_width = DIGIT_SIZE

        new_height = max(
            1,
            int(
                height
                * DIGIT_SIZE
                / width
            )
        )

    else:

        new_height = DIGIT_SIZE

        new_width = max(
            1,
            int(
                width
                * DIGIT_SIZE
                / height
            )
        )

    image = image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    # ========================================================
    # Create 28 × 28 canvas
    # ========================================================

    canvas = Image.new(
        "L",
        (
            MNIST_SIZE,
            MNIST_SIZE
        ),
        0
    )

    # ========================================================
    # Initial placement
    # ========================================================

    x = (
        MNIST_SIZE
        - new_width
    ) // 2

    y = (
        MNIST_SIZE
        - new_height
    ) // 2

    canvas.paste(
        image,
        (
            x,
            y
        )
    )

    # ========================================================
    # Calculate center of mass
    # on the 28 × 28 image
    # ========================================================

    center_x, center_y = (
        calculate_center_of_mass(
            canvas
        )
    )

    print(
        f"Center of mass after resize: "
        f"({center_x:.2f}, {center_y:.2f})"
    )

    # ========================================================
    # Desired MNIST center
    # ========================================================

    target_x = (
        MNIST_SIZE - 1
    ) / 2

    target_y = (
        MNIST_SIZE - 1
    ) / 2

    # ========================================================
    # Calculate required shift
    # ========================================================

    shift_x = (
        target_x - center_x
    )

    shift_y = (
        target_y - center_y
    )

    print(
        f"Center-of-mass shift: "
        f"({shift_x:.2f}, {shift_y:.2f})"
    )

    # ========================================================
    # Shift image
    # ========================================================

    canvas = shift_image(
        canvas,
        shift_x,
        shift_y
    )

    return canvas


# ============================================================
# MAIN
# ============================================================

def main():

    image_path = input(
        "Enter image path: "
    ).strip()

    processed = preprocess_image(
        image_path
    )

    # ========================================================
    # Save processed image
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/"
        "processed_digit_centered.png"
    )

    processed.save(
        output_path
    )

    print()

    print(
        "Processed image saved to:"
    )

    print(
        output_path
    )

    # ========================================================
    # Display
    # ========================================================

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        processed,
        cmap="gray"
    )

    plt.title(
        "Center-of-Mass Aligned Digit"
    )

    plt.axis(
        "off"
    )

    plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()


