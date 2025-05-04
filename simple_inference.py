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
from sklearn.metrics import roc_auc_score

CATEGORY_LIST = ["CNV", "DME", "DRUSEN", "NORMAL"]

all_scores = []
all_labels = []

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
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))

    for i in range(3):
        x = images[i].unsqueeze(0).to(model.device)
        embedding_loss, x_hat, _, alm_map, encoding_indices = model(x)

        # Residual map
        residual_map = torch.abs(x - x_hat)
        res_np = residual_map[0, 0].detach().cpu().numpy()
        res_norm = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)
        res_score = res_norm.mean()
        res_pred = 1 if res_score >= 0.5 else 0

        # ALM map
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear')
        alm_np = alm_up[0, 0].detach().cpu().numpy()
        alm_norm = (alm_np - alm_np.min()) / (alm_np.max() - alm_np.min() + 1e-8)
        alm_score = alm_np.mean()
        alm_pred = 1 if alm_score >= 0.5 else 0

        # Ground-truth label
        label = 0 if category_name == 'NORMAL' else 1

        try:
            auroc_alm = roc_auc_score([label, 1 - label], [alm_score, 0.0])
            auroc_res = roc_auc_score([label, 1 - label], [res_score, 0.0])
        except:
            auroc_alm = auroc_res = -1  # fallback

        alm_str = (
            f"ALM\nScore={alm_score:.2f} → Pred: {alm_pred} ({'Abnormal' if alm_pred else 'Normal'})"
            f"\nAUROC={auroc_alm:.4f}"
        )
        res_str = (
            f"Residual\nScore={res_score:.2f} → Pred: {res_pred} ({'Abnormal' if res_pred else 'Normal'})"
            f"\nAUROC={auroc_res:.4f}"
        )
        imgs = [x[0, 0], x_hat[0, 0], alm_norm, res_norm]
        titles = ['Input', 'Reconstruction', alm_str, res_str]

        for j in range(4):
            ax = axes[i, j]
            if j >= 2:
                ax.imshow(imgs[j], cmap='jet')  # Color for ALM/Residual
            else:
                ax.imshow(imgs[j].detach().cpu(), cmap='gray')
            ax.set_title(titles[j], fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{category_name}_grid.png")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    
    model_ckpt = Path(args.model_dir)
    ckpt_dir = model_ckpt.parent
    config_path = Path(ckpt_dir) / "config.yaml"
    output_dir = os.path.join(ckpt_dir, 'inference')
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = Args(**args_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitVQVAE.load_from_checkpoint(model_ckpt, args=model_args)
    model.to(device).eval()

    for category in CATEGORY_LIST:
        images, labels = load_category_samples(category)
        visualize_category(model, images, labels, category, output_dir)

    print(f"All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
