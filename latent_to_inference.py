# save_latents.py
import sys
import os
import torch
from models.vqvae import LitVQVAE
from utils import save_latents_from_model
from datasets.dataset import get_dataloaders
from argparse import ArgumentParser
from pathlib import Path
import subprocess
import yaml

def main(save_dir):
    # 1. args 복원 (config.yaml로부터)
    config_path = Path(save_dir).parent / "config.yaml"
    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args = Args(**args_dict)

    # 2. 모델 로드
    ckpt_path = Path(save_dir)
    model = LitVQVAE.load_from_checkpoint(ckpt_path, args=args)
    model.eval().cuda()

    # 3. train_loader 가져오기 (normal data 기준)
    train_loader, _ = get_dataloaders(args)

    # 4. latent 저장
    latent_path = Path(save_dir).parent / "latent" / "latent_e_indices.npy"
    os.makedirs(latent_path.parent, exist_ok=True)
    save_latents_from_model(model, train_loader, args.embedding_dim, save_path=latent_path)

    # # 5. PixelCNN 학습
    # pixelcnn_save_dir = Path(save_dir) / "pixelcnn"
    # os.makedirs(pixelcnn_save_dir, exist_ok=True)

    # print("Running PixelCNN command:")
    # print([
    #     "python", "pixelcnn/gated_pixelcnn.py",
    #     "--data_path", str(latent_path),
    #     "--dataset", "LATENT_BLOCK",
    #     "--save_dir", str(pixelcnn_save_dir),
    #     "--save"
    # ])

    # subprocess.run([
    #     "python", "pixelcnn/gated_pixelcnn.py",
    #     "--data_path", str(latent_path),
    #     "--dataset", "LATENT_BLOCK",
    #     "--save_dir", str(Path(pixelcnn_save_dir).resolve()),
    #     "--save"
    # ], stdout=sys.stdout, stderr=sys.stderr)

    # # 6. Inference 실행
    # subprocess.run([
    #     "python", "inference.py",
    #     "--model_ckpt", str(ckpt_path),
    #     "--pixelcnn_ckpt", str(pixelcnn_save_dir / "best_model.pt"),
    #     "--output_dir", str(Path(save_dir) / "inference")
    # ])

    subprocess.run([
        "python", "simple_inference.py",
        "--model_dir", ckpt_path
    ])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Original save directory (where config.yaml is)")
    args = parser.parse_args()

    main(args.save_dir)
