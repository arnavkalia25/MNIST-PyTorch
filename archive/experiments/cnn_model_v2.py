
import torch.nn as nn


class MNISTCNNv2(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # Convolutional Feature Extractor
        # ====================================================

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU()
        )

        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                10
            )
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = self.classifier(x)

        return x
