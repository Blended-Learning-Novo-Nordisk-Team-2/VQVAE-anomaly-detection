import os
import argparse
import torch
import numpy as np
from pathlib import Path

from models.vqvae import LitVQVAE
from utils import save_latents_from_model
from datasets.dataset import get_dataloaders


def main(args):
    # 1. Load trained Lightning checkpoint
    print("🔄 Loading model from:", args.ckpt_path)
    lit_model = LitVQVAE.load_from_checkpoint(checkpoint_path=args.ckpt_path, args=args)
    vqvae_model = lit_model.model.eval()

    # 2. Load dataset & dataloader using training-compatible function
    print("📦 Loading dataloader (train split)...")
    train_loader, _ = get_dataloaders(args)

    # 3. Create save directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 4. Save latent indices
    print(f"💾 Saving latent indices to {args.save_path}...")
    save_latents_from_model(
        model=vqvae_model,
        dataloader=train_loader,
        embedding_dim=None,  # kept for compatibility, ignored in function
        save_path=args.save_path
    )
    print("✅ Latent saving complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to .ckpt file')
    parser.add_argument('--save_path', type=str, default='latent/latent_e_indices.npy', help='Path to save .npy file')

    # Add full training-compatible arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_hiddens", type=int, default=256)
    parser.add_argument("--n_residual_hiddens", type=int, default=64)
    parser.add_argument("--n_residual_layers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--n_embeddings", type=int, default=128)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--save", type=bool, default=True)

    args = parser.parse_args()
    main(args)