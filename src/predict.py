
import argparse
import os

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps

from torchvision import transforms

from model import MNISTModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_mlp.pth"

IMAGE_SIZE = 28

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = MNISTModel()

    model = model.to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    )

    print(
        f"Original image size: {image.size}"
    )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # Convert to black background / white digit
    # --------------------------------------------------------

    # Find minimum and maximum pixel values
    min_value, max_value = image.getextrema()

    # If the image has a white background and dark digit,
    # invert it so it looks like MNIST:
    #
    # MNIST:
    # black background = 0
    # white digit      = higher values
    #
    if min_value < 100 and max_value > 200:

        # Determine whether background is mostly white
        center_pixel = image.getpixel(
            (
                image.width // 2,
                image.height // 2
            )
        )

        if center_pixel > 180:

            image = ImageOps.invert(
                image
            )

    # --------------------------------------------------------
    # Auto-crop empty background
    # --------------------------------------------------------

    # Threshold the image
    threshold = 30

    binary = image.point(
        lambda p: 255 if p > threshold else 0
    )

    bbox = binary.getbbox()

    if bbox is not None:

        image = image.crop(
            bbox
        )

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    width, height = image.size

    if width == 0 or height == 0:

        raise ValueError(
            "Invalid image dimensions."
        )

    # Leave some padding around the digit
    target_size = 20

    if width > height:

        new_width = target_size

        new_height = int(
            height * target_size / width
        )

    else:

        new_height = target_size

        new_width = int(
            width * target_size / height
        )

    new_width = max(
        1,
        new_width
    )

    new_height = max(
        1,
        new_height
    )

    image = image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create 28x28 black canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        0
    )

    # --------------------------------------------------------
    # Center the digit
    # --------------------------------------------------------

    x = (
        IMAGE_SIZE - new_width
    ) // 2

    y = (
        IMAGE_SIZE - new_height
    ) // 2

    canvas.paste(
        image,
        (
            x,
            y
        )
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    transform = transforms.ToTensor()

    image_tensor = transform(
        canvas
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(
        0
    )

    return image_tensor


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    image_tensor
):

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        predicted_digit = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_digit
        ].item()

    return (
        predicted_digit,
        confidence,
        probabilities
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    predicted_digit,
    confidence,
    probabilities
):

    print()

    print(
        "===== PREDICTION ====="
    )

    print(
        f"Predicted digit: "
        f"{predicted_digit}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print()

    print(
        "===== CLASS PROBABILITIES ====="
    )

    for digit in range(10):

        probability = probabilities[
            0,
            digit
        ].item()

        print(
            f"Digit {digit}: "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Predict a handwritten digit "
            "using the trained MNIST MLP."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to the digit image."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"Using device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found: "
            f"{MODEL_PATH}\n"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    image_tensor = preprocess_image(
        args.image
    )

    print(
        f"Processed image shape: "
        f"{image_tensor.shape}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    (
        predicted_digit,
        confidence,
        probabilities
    ) = predict(
        model,
        image_tensor
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results(
        predicted_digit,
        confidence,
        probabilities
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

