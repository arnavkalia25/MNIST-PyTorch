import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_mnist_cnn_augmented.pth"

IMAGE_SIZE = 28

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CNN Model
# ============================================================

class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model = MNISTCNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# Improved Image Preprocessing
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    original_array = np.array(image)

    # --------------------------------------------------------
    # Automatically determine foreground/background
    # --------------------------------------------------------

    if original_array.mean() > 127:

        image = ImageOps.invert(image)

    image_array = np.array(image)

    # --------------------------------------------------------
    # Remove very weak background pixels
    # --------------------------------------------------------

    threshold = 30

    binary = image_array > threshold

    # --------------------------------------------------------
    # Find digit bounding box
    # --------------------------------------------------------

    coords = np.argwhere(binary)

    if coords.size == 0:

        # Empty image fallback
        processed = Image.new(
            "L",
            (28, 28),
            0
        )

        tensor = transforms.ToTensor()(
            processed
        ).unsqueeze(0)

        return processed, tensor

    y_min, x_min = coords.min(axis=0)

    y_max, x_max = coords.max(axis=0)

    # --------------------------------------------------------
    # Crop digit
    # --------------------------------------------------------

    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    width, height = cropped.size

    padding = int(
        max(width, height) * 0.20
    )

    cropped = ImageOps.expand(
        cropped,
        border=padding,
        fill=0
    )

    # --------------------------------------------------------
    # Make image square
    # --------------------------------------------------------

    width, height = cropped.size

    side = max(
        width,
        height
    )

    square = Image.new(
        "L",
        (side, side),
        0
    )

    left = (side - width) // 2
    top = (side - height) // 2

    square.paste(
        cropped,
        (left, top)
    )

    # --------------------------------------------------------
    # Resize to 28 × 28
    # --------------------------------------------------------

    processed = square.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Center of mass
    # --------------------------------------------------------

    processed_array = np.array(
        processed,
        dtype=np.float32
    )

    total = processed_array.sum()

    if total > 0:

        y_indices, x_indices = np.indices(
            processed_array.shape
        )

        center_x = (
            (x_indices * processed_array).sum()
            / total
        )

        center_y = (
            (y_indices * processed_array).sum()
            / total
        )

        target_center = 13.5

        shift_x = int(
            round(target_center - center_x)
        )

        shift_y = int(
            round(target_center - center_y)
        )

        processed = ImageOps.expand(
            processed,
            border=20,
            fill=0
        )

        processed = processed.transform(
            processed.size,
            Image.AFFINE,
            (
                1,
                0,
                -shift_x,
                0,
                1,
                -shift_y
            ),
            resample=Image.Resampling.BILINEAR
        )

        processed = processed.crop(
            (20, 20, 48, 48)
        )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    tensor = transforms.ToTensor()(
        processed
    ).unsqueeze(0)

    return processed, tensor


# ============================================================
# Prediction
# ============================================================

def predict(model, image_tensor):

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_digit = prediction.item()

    confidence_value = confidence.item()

    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )

    return (
        predicted_digit,
        confidence_value,
        probabilities
    )


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# Header
# ============================================================

st.title(
    "🔢 MNIST Handwritten Digit Classifier"
)

st.write(
    """
    Upload a handwritten digit image and the trained
    CNN will predict the digit.
    """
)


# ============================================================
# Model Information
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Model:** CNN + Data Augmentation"
    )

    st.write(
        "**Dataset:** MNIST"
    )

    st.write(
        "**Input Size:** 28 × 28"
    )

    st.write(
        "**Classes:** 10 digits (0–9)"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.write(
        "**Test Accuracy:** 99.26%"
    )


# ============================================================
# Load Model
# ============================================================

try:

    model = load_model()

    st.success(
        "CNN model loaded successfully."
    )

except Exception as e:

    st.error(
        f"Failed to load model: {e}"
    )

    st.stop()


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# Process Image
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    st.subheader(
        "Original Image"
    )

    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_image, image_tensor = (
        preprocess_image(image)
    )

    # --------------------------------------------------------
    # Show processed image
    # --------------------------------------------------------

    st.subheader(
        "Processed Image"
    )

    st.image(
        processed_image,
        caption="Final 28 × 28 CNN input",
        width=250
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_digit, confidence, probabilities = (
        predict(
            model,
            image_tensor
        )
    )

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Prediction"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Digit",
            str(predicted_digit)
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Confidence warning
    # --------------------------------------------------------

    if confidence < 0.50:

        st.warning(
            "The model has low confidence. "
            "Try uploading a clearer, centered digit."
        )

    elif confidence < 0.80:

        st.info(
            "The model is moderately confident "
            "in this prediction."
        )

    else:

        st.success(
            "The model is highly confident "
            "in this prediction."
        )

    # --------------------------------------------------------
    # Probability chart
    # --------------------------------------------------------

    st.subheader(
        "Class Probabilities"
    )

    digits = list(range(10))

    fig, ax = plt.subplots()

    ax.bar(
        digits,
        probabilities * 100
    )

    ax.set_xlabel(
        "Digit"
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    ax.set_title(
        "CNN Prediction Probabilities"
    )

    ax.set_xticks(
        digits
    )

    st.pyplot(fig)

    # --------------------------------------------------------
    # Detailed probabilities
    # --------------------------------------------------------

    st.subheader(
        "Detailed Probabilities"
    )

    for digit, probability in zip(
        digits,
        probabilities
    ):

        st.write(
            f"Digit {digit}: "
            f"{probability * 100:.2f}%"
        )

else:

    st.info(
        "Upload a handwritten digit image to begin."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "MNIST-PyTorch • CNN + Data Augmentation"
)