
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torchvision import transforms

from cnn_model import MNISTCNN


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_mnist_cnn.pth"

RESULTS_DIR = "results"

MNIST_SIZE = 28
DIGIT_SIZE = 20


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CENTER OF MASS
# ============================================================

def calculate_center_of_mass(image):

    array = np.asarray(
        image,
        dtype=np.float32
    )

    total_mass = array.sum()

    if total_mass == 0:

        return (
            image.width / 2,
            image.height / 2
        )

    y_indices, x_indices = np.indices(
        array.shape
    )

    center_x = (
        (x_indices * array).sum()
        / total_mass
    )

    center_y = (
        (y_indices * array).sum()
        / total_mass
    )

    return center_x, center_y


# ============================================================
# SHIFT IMAGE
# ============================================================

def shift_image(
    image,
    shift_x,
    shift_y
):

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

def preprocess_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"\nImage not found:\n{image_path}"
        )

    # ========================================================
    # 1. Load original image
    # ========================================================

    original = Image.open(
        image_path
    )

    print(
        f"Original image size: "
        f"{original.size}"
    )

    # ========================================================
    # 2. Convert to grayscale
    # ========================================================

    grayscale = original.convert(
        "L"
    )

    # ========================================================
    # 3. Calculate brightness
    # ========================================================

    image_array = np.asarray(
        grayscale,
        dtype=np.float32
    )

    mean_pixel = image_array.mean()

    print(
        f"Average pixel value: "
        f"{mean_pixel:.2f}"
    )

    # ========================================================
    # 4. Invert if background is bright
    # ========================================================

    if mean_pixel > 127:

        grayscale = ImageOps.invert(
            grayscale
        )

        print(
            "Image inverted."
        )

    # ========================================================
    # 5. Threshold
    # ========================================================

    threshold = 40

    binary = grayscale.point(
        lambda pixel:
        255 if pixel > threshold
        else 0
    )

    # ========================================================
    # 6. Detect bounding box
    # ========================================================

    bbox = binary.getbbox()

    if bbox is None:

        raise ValueError(
            "Could not detect a digit."
        )

    print(
        f"Detected bounding box: "
        f"{bbox}"
    )

    # ========================================================
    # 7. Crop digit
    # ========================================================

    cropped = grayscale.crop(
        bbox
    )

    print(
        f"Cropped digit size: "
        f"{cropped.size}"
    )

    # ========================================================
    # 8. Add padding
    # ========================================================

    width, height = cropped.size

    padding = int(
        max(width, height) * 0.15
    )

    padded_width = (
        width + 2 * padding
    )

    padded_height = (
        height + 2 * padding
    )

    padded = Image.new(
        "L",
        (
            padded_width,
            padded_height
        ),
        0
    )

    padded.paste(
        cropped,
        (
            padding,
            padding
        )
    )

    # ========================================================
    # 9. Resize while preserving aspect ratio
    # ========================================================

    width, height = padded.size

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

    resized = padded.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    # ========================================================
    # 10. Place on 28 × 28 canvas
    # ========================================================

    canvas = Image.new(
        "L",
        (
            MNIST_SIZE,
            MNIST_SIZE
        ),
        0
    )

    x = (
        MNIST_SIZE - new_width
    ) // 2

    y = (
        MNIST_SIZE - new_height
    ) // 2

    canvas.paste(
        resized,
        (
            x,
            y
        )
    )

    # ========================================================
    # 11. Calculate center of mass
    # ========================================================

    center_x, center_y = (
        calculate_center_of_mass(
            canvas
        )
    )

    print(
        f"Center of mass: "
        f"({center_x:.2f}, "
        f"{center_y:.2f})"
    )

    # ========================================================
    # 12. Target center
    # ========================================================

    target_x = (
        MNIST_SIZE - 1
    ) / 2

    target_y = (
        MNIST_SIZE - 1
    ) / 2

    # ========================================================
    # 13. Calculate shift
    # ========================================================

    shift_x = (
        target_x - center_x
    )

    shift_y = (
        target_y - center_y
    )

    print(
        f"Center-of-mass shift: "
        f"({shift_x:.2f}, "
        f"{shift_y:.2f})"
    )

    # ========================================================
    # 14. Center image
    # ========================================================

    centered = shift_image(
        canvas,
        shift_x,
        shift_y
    )

    # ========================================================
    # 15. Convert to tensor
    # ========================================================

    transform = transforms.ToTensor()

    tensor = transform(
        centered
    )

    # ========================================================
    # 16. Add batch dimension
    # ========================================================

    tensor = tensor.unsqueeze(
        0
    )

    # ========================================================
    # 17. Move to device
    # ========================================================

    tensor = tensor.to(
        device
    )

    # ========================================================
    # Return all stages
    # ========================================================

    stages = {
        "original": original,
        "grayscale": grayscale,
        "binary": binary,
        "cropped": cropped,
        "resized": resized,
        "final": centered
    }

    return tensor, stages


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = MNISTCNN()

    model = model.to(
        device
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    image_tensor
):

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_digit = (
        prediction.item()
    )

    confidence_percentage = (
        confidence.item() * 100
    )

    probabilities_percentage = (
        probabilities[0]
        .cpu()
        .numpy()
        * 100
    )

    return (
        predicted_digit,
        confidence_percentage,
        probabilities_percentage
    )


# ============================================================
# SAVE PREPROCESSED IMAGE
# ============================================================

def save_final_image(
    image,
    image_path
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    filename = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        RESULTS_DIR,
        f"processed_{filename}.png"
    )

    image.save(
        output_path
    )

    print()

    print(
        f"Final 28x28 image saved to:"
    )

    print(
        output_path
    )


# ============================================================
# VISUALIZE PREPROCESSING
# ============================================================

def visualize_stages(
    stages,
    predicted_digit,
    confidence,
    image_path
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    filename = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        RESULTS_DIR,
        f"preprocessing_{filename}.png"
    )

    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    # ========================================================
    # Stage names
    # ========================================================

    stage_info = [
        (
            "Original",
            stages["original"]
        ),
        (
            "Grayscale",
            stages["grayscale"]
        ),
        (
            "Threshold",
            stages["binary"]
        ),
        (
            "Cropped",
            stages["cropped"]
        ),
        (
            "Resized",
            stages["resized"]
        ),
        (
            "Final 28×28",
            stages["final"]
        )
    ]

    # ========================================================
    # Display images
    # ========================================================

    for axis, (title, image) in zip(
        axes.flat,
        stage_info
    ):

        axis.imshow(
            image,
            cmap="gray"
        )

        axis.set_title(
            title
        )

        axis.axis(
            "off"
        )

    # ========================================================
    # Overall title
    # ========================================================

    fig.suptitle(
        (
            f"CNN Prediction: "
            f"{predicted_digit} "
            f"({confidence:.2f}% confidence)"
        ),
        fontsize=16
    )

    plt.tight_layout()

    # ========================================================
    # Save
    # ========================================================

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Preprocessing visualization saved to:"
    )

    print(
        output_path
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Argument parser
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "MNIST CNN custom image "
            "prediction"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model()

    print(
        "CNN model loaded successfully."
    )

    # ========================================================
    # Preprocess
    # ========================================================

    (
        image_tensor,
        stages
    ) = preprocess_image(
        args.image
    )

    print(
        f"Processed image shape: "
        f"{image_tensor.shape}"
    )

    # ========================================================
    # Prediction
    # ========================================================

    (
        predicted_digit,
        confidence,
        probabilities
    ) = predict(
        model,
        image_tensor
    )

    # ========================================================
    # Display prediction
    # ========================================================

    print()

    print(
        "===== CNN PREDICTION ====="
    )

    print(
        f"Predicted digit: "
        f"{predicted_digit}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    # ========================================================
    # Display probabilities
    # ========================================================

    print()

    print(
        "===== CLASS PROBABILITIES ====="
    )

    for digit in range(10):

        print(
            f"Digit {digit}: "
            f"{probabilities[digit]:.2f}%"
        )

    # ========================================================
    # Save final 28 × 28 image
    # ========================================================

    save_final_image(
        stages["final"],
        args.image
    )

    # ========================================================
    # Save preprocessing visualization
    # ========================================================

    visualize_stages(
        stages,
        predicted_digit,
        confidence,
        args.image
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
