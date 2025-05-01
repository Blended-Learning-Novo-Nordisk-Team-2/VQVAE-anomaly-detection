import os
from datetime import datetime
import yaml
import torch
import wandb
import subprocess
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.vqvae import LitVQVAE
from dataset import get_dataloaders
import argparse
from utils import save_latents_from_model
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVED_MODELS = ROOT / "saved_models"

def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--save", type=bool, default=True, help="Save outputs after training")
    return parser.parse_args()

def main():
    args = parse_args()

    seed_everything(args.seed)
    run_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = SAVED_MODELS / run_time_str
    save_dir.mkdir(parents=True, exist_ok=True)



    model = LitVQVAE(args)
    train_loader, val_loader = get_dataloaders(args)

    checkpoint = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=save_dir,
        filename="best-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True
    )

    early_stop = EarlyStopping(
        monitor="val_total_loss",
        patience=args.patience,
        min_delta=args.min_delta,
        mode="min",
    )

    logger = WandbLogger(
        project="VQVAE-sweep",
        name=f"latent{args.embedding_dim}_lr{args.learning_rate}",
        save_dir=save_dir,
        log_model=True
    )

    logger.experiment.config.update(vars(args), allow_val_change=True)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.device == "gpu" else "cpu",
        logger=logger,
        callbacks=[checkpoint, early_stop],
        default_root_dir=save_dir,
        log_every_n_steps=10,
        devices=1,
        precision='16-mixed'
    )

    trainer.fit(model, train_loader, val_loader)

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    if args.save:
        # 1. latent 저장
        latent_path = os.path.join(save_dir, "latent", "latent_e_indices.npy")
        os.makedirs(os.path.dirname(latent_path), exist_ok=True)
        save_latents_from_model(model, train_loader, args.embedding_dim, save_path=latent_path)

if __name__ == "__main__":
    main()
