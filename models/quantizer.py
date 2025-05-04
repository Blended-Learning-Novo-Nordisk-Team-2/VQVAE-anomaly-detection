import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE that discretizes continuous latent vectors.
    
    This module implements the discretization bottleneck of VQ-VAE by:
    1. Maintaining a codebook of embedding vectors
    2. Finding the closest embedding for each input vector
    3. Computing commitment loss to ensure encoder outputs stay close to codebook
    
    Parameters:
    - n_e: Number of embeddings in codebook
    - e_dim: Dimension of each embedding vector
    - beta: Commitment cost coefficient for loss term
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # Initialize embedding table (codebook)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # Initialize embeddings uniformly
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Quantize continuous latent vectors to discrete codebook entries.
        
        Process:
        1. Reshape input to (batch*height*width, channel)
        2. Compute distances to all codebook entries
        3. Find closest codebook entry for each vector
        4. Compute commitment loss
        5. Return quantized vectors and metrics
        
        Args:
            z: Continuous latent vectors from encoder (B,C,H,W)
            
        Returns:
            loss: Commitment loss
            z_q: Quantized vectors
            perplexity: Codebook usage metric
            encoding_indices: Indices of used codebook entries
        """
        device = z.device
        # Reshape input for distance computation
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Compute distances to all codebook entries
        # (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # Find closest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get quantized vectors from codebook
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute commitment loss:
        # 1. Codebook loss: ||z_q.detach() - z||^2
        # 2. Commitment loss: beta * ||z_q - z.detach()||^2
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # Straight-through estimator: preserve gradients
        z_q = z + (z_q - z).detach()

        # Compute perplexity (measure of codebook usage)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape output to match input dimensions
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Get encoding indices in original shape
        B, H, W, _ = z.shape
        encoding_indices = min_encoding_indices.view(B, H, W)

        return loss, z_q, perplexity, encoding_indices
