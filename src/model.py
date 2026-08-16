import torch.nn as nn


class MNISTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # =========================
            # Input
            # =========================

            nn.Flatten(),

            # =========================
            # Layer 1
            # =========================

            nn.Linear(
                28 * 28,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            # =========================
            # Layer 2
            # =========================

            nn.Linear(
                256,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            # =========================
            # Layer 3
            # =========================

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            # =========================
            # Output Layer
            # =========================

            nn.Linear(
                64,
                10
            )
        )

    def forward(self, x):

        return self.network(x)