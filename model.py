

import torch.nn as nn
import torch.nn.functional as F


# ==============================
# Generator for MNIST (28x28, 1 channel)
# =============================
class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # Fully connected layer to reshape z into feature map
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)   # Shape: (batch, 256*7*7)

        # Step 2: First unsampling block
        # Input: (batch, 256, 7, 7)
        # ConvTranspose2d doubles spatial size → (batch, 128, 14, 14)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),  # Normalization to stabilize training
            nn.ReLU(True)                         # Non-linearity
        )

        # Step 3: Second upsampling block
        # Input: (batch, 128, 14, 14)
        # ConvTransportation doubles spatial size → (batch, 64, 28, 28)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        )

        # Step 4: Final convolution
        # Input: (batch, 64, 28, 28)
        # Output: (batch, 1, 28, 28) → grayscale MNIST images
        self.final = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # Step 5: Tanh activation
        # Ensures pixel values are in range [-1, 1]
        self.tanh = nn.Tanh()

    def forward(self, z):

        # Pass noise vector through fully connected layer
        x = self.fc(z)   # Shape: (batch, 256*7*7)

        # Reshape into feature map for ConvTranspose2d
        x = x.view(-1, 256, 7, 7)   # Shape: (batch, 256, 7, 7)

        # Apply upsampling blocks
        x = self.block1(x)   # → (batch, 128, 14, 14)
        x = self.block2(x)   # → (batch, 64, 28, 28)

        # Final image output
        x = self.final(x)  # → (batch, 1, 28, 28)
        return self.tanh(x) # → scaled to [-1, 1]


# ==============================
# Discriminator (Critic) for MNIST
# ==============================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Step 1: Input is (batch, 1, 28, 28)
        # Conv layer downsamples → (batch, 64, 14, 14)
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Step 2: Downsample again → (batch, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Step 3: Downsample once more → (batch, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Step 4: Final convolution to single value
        # Input: (batch, 256, 4, 4)
        # Output: (batch, 1, 1, 1)
        self.final = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0) # -> 1×1 output

    def forward(self, x):
      # Apply convolutional blocks
        x = self.main(x)  # → (batch, 256, 4, 4)
        x = self.final(x) # → (batch, 1, 1, 1)

        # Flatten to shape (batch,) → one score per image
        return x.view(-1)



