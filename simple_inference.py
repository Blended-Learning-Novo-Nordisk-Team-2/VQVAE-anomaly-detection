import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vqvae import LitVQVAE
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from glob import glob
import torch.nn.functional as F
from pathlib import Path
import yaml

CATEGORY_LIST = ["CNV", "DME", "DRUSEN", "NORMAL"]

def load_category_samples(category, num_samples=3, base_dir='test_dataset'):
    """
    Load num_samples images from a flat directory matching a category prefix (e.g., CNV, DME, etc.)
    Assumes images are named like 'CNV-xxxx.jpeg', etc.
    """
    transform = Compose([
        Resize((512, 512)),
        ToTensor()
    ])

    pattern = os.path.join(base_dir, f"{category}-*.jpeg")
    img_paths = sorted(glob(pattern))[:num_samples]

    images = []
    labels = []

    for path in img_paths:
        img = Image.open(path).convert("L")
        images.append(transform(img))
        labels.append(os.path.basename(path))

    return torch.stack(images), labels

def visualize_category(model, images, labels, category_name, save_dir):
    model.eval()
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))  # reduced to 4 images per row

    for i in range(3):
        x = images[i].unsqueeze(0).to(model.device)
        embedding_loss, x_hat, _, alm_map, encoding_indices = model(x)
        residual_map = torch.abs(x - x_hat)
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear')
        res_up = residual_map

        imgs = [x[0,0], x_hat[0,0], alm_up[0,0], res_up[0,0]]
        titles = ['Input', 'Reconstruction', 'ALM', 'Residual']

        for j in range(4):
            ax = axes[i, j]
            ax.imshow(imgs[j].detach().cpu(), cmap='gray')
            ax.set_title(titles[j])
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{category_name}_grid.png")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    
    ckpt_dir = os.path.join(args.model_dir, '2025-05-02_22-48-04')
    config_path = Path(ckpt_dir) / "config.yaml"
    output_dir = os.path.join(ckpt_dir, 'inference')
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = Args(**args_dict)
    model_ckpt = Path(ckpt_dir) / 'best-epoch=99-val_total_loss=0.0537.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitVQVAE.load_from_checkpoint(model_ckpt, args=model_args)
    model.to(device).eval()

    for category in CATEGORY_LIST:
        images, labels = load_category_samples(category)
        visualize_category(model, images, labels, category, output_dir)

    print(f"All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
