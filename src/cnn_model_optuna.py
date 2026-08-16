import torch.nn as nn


class MNISTCNNOptuna(nn.Module):

    def __init__(
        self,
        conv1_channels=32,
        conv2_channels=64,
        hidden_size=128,
        dropout=0.3
    ):

        super().__init__()

        # ====================================================
        # Feature Extractor
        # ====================================================

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Convolution Block 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Convolution Block 2
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        # ====================================================
        # Classifier
        # ====================================================

        # MNIST:
        #
        # 28 x 28
        # ↓ MaxPool
        # 14 x 14
        # ↓ MaxPool
        # 7 x 7
        #
        # Therefore:
        #
        # conv2_channels × 7 × 7

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                conv2_channels * 7 * 7,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x