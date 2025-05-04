import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    Decoder network (p_phi) that maps latent vectors z back to original space x.
    
    Architecture:
    - Series of transposed convolutional layers with decreasing channels
    - Residual blocks for better feature reconstruction
    - Final layer maps to single channel output
    
    Parameters:
    - in_dim: Input channel dimension
    - h_dim: Hidden layer dimension
    - res_h_dim: Hidden dimension in residual blocks
    - n_res_layers: Number of residual layers
    - dropout: Dropout rate for regularization
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, dropout):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            # First deconv block: in_dim -> h_dim
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            # Residual stack for better feature reconstruction
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Dropout(dropout),

            # Second deconv block: h_dim -> h_dim
            nn.ConvTranspose2d(h_dim, h_dim,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third deconv block: h_dim -> h_dim/2
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Fourth deconv block: h_dim/2 -> h_dim/4
            nn.ConvTranspose2d(h_dim // 2, h_dim // 4,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),

            # Final deconv block: h_dim/4 -> 1 (output channel)
            nn.ConvTranspose2d(h_dim // 4, 1, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # Test with random input data
    x = np.random.random_sample((3, 256, 64, 64))
    x = torch.tensor(x).float()

    # Test decoder forward pass
    decoder = Decoder(256, 256, 1, 64)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
