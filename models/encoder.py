import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    Encoder network (q_theta) that maps input data x to latent space z.
    For VQ-VAE, this network outputs parameters of a categorical distribution.
    
    Architecture:
    - Series of convolutional layers with increasing channels
    - Residual blocks for better gradient flow
    - Adaptive pooling to ensure fixed output size
    
    Parameters:
    - in_dim: Input channel dimension
    - h_dim: Hidden layer dimension
    - res_h_dim: Hidden dimension in residual blocks
    - n_res_layers: Number of residual layers
    - dropout: Dropout rate for regularization
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, dropout):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            # First conv block: in_dim -> h_dim/4
            nn.Conv2d(in_dim, h_dim // 4, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second conv block: h_dim/4 -> h_dim/2
            nn.Conv2d(h_dim // 4, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Third conv block: h_dim/2 -> h_dim
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Fourth conv block: h_dim -> h_dim (reduced kernel)
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Fifth conv block: h_dim -> h_dim (reduced kernel)
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            # Residual stack for better feature extraction
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers),
            # Adaptive pooling to ensure fixed output size
            nn.AdaptiveAvgPool2d((64, 64))
        )

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    # Test with random input data
    x = np.random.random_sample((3, 1, 512, 512))
    x = torch.tensor(x).float()
    print('Encoder in shape:', x.shape)

    # Test encoder forward pass
    encoder = Encoder(1, 256, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
