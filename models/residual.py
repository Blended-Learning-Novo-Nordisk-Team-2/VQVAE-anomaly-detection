import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    """
    Single residual layer implementation.
    
    Architecture:
    - Two convolutional layers with ReLU activation
    - Skip connection that adds input to output
    - Helps with gradient flow in deep networks
    
    Parameters:
    - in_dim: Input channel dimension
    - h_dim: Output channel dimension
    - res_h_dim: Hidden dimension in residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            # First conv block with 3x3 kernel
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            # Second conv block with 1x1 kernel
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        # Skip connection: add input to output
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    Stack of residual layers for deep feature extraction.
    
    Architecture:
    - Multiple residual layers in sequence
    - Final ReLU activation
    - Used in both encoder and decoder
    
    Parameters:
    - in_dim: Input channel dimension
    - h_dim: Output channel dimension
    - res_h_dim: Hidden dimension in residual blocks
    - n_res_layers: Number of residual layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        # Create stack of identical residual layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        # Pass through each residual layer
        for layer in self.stack:
            x = layer(x)
        # Final ReLU activation
        x = F.relu(x)
        return x


if __name__ == "__main__":
    # Test with random input data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    
    # Test single residual layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    
    # Test residual stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
