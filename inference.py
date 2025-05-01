import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vqvae import LitVQVAE
from pixelcnn.models import GatedPixelCNN
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from glob import glob
import torch.nn.functional as F
from pathlib import Path
import yaml

CATEGORY_LIST = ["CNV", "DME", "DRUSEN", "NORMAL"]

def load_pixelcnn_model(weight_path, device='cpu'):
    input_dim = 128
    dim = 64
    n_layers = 15
    model = GatedPixelCNN(input_dim=input_dim, dim=dim, n_layers=n_layers, n_classes=1).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_nll_map(model, latent_map):
    x = latent_map.unsqueeze(0).long()
    label = torch.zeros(1, dtype=torch.long, device=x.device)
    logits = model(x, label)
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-9)
    true_class = x.unsqueeze(1)
    log_prob = log_probs.gather(1, true_class).squeeze(1)
    return -log_prob.detach().cpu().squeeze(0)

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
    img_paths = sorted(glob(pattern))[:num_samples]  # grab first N

    images = []
    labels = []

    for path in img_paths:
        img = Image.open(path).convert("L")
        images.append(transform(img))  # [1, H, W]
        labels.append(os.path.basename(path))

    return torch.stack(images), labels

def visualize_category(model, pixelcnn, images, labels, category_name, save_dir):
    model.eval(); pixelcnn.eval()
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))
    for i in range(3):
        x = images[i].unsqueeze(0).to(model.device)
        embedding_loss, x_hat, _, alm_map, encoding_indices = model(x, return_extra=True)
        residual_map = torch.abs(x - x_hat)
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear')
        res_up = residual_map
        nll_map = get_nll_map(pixelcnn, encoding_indices[0])
        nll_up = F.interpolate(nll_map.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear')[0,0]
        imgs = [x[0,0], x_hat[0,0], alm_up[0,0], encoding_indices[0].float(), nll_up, res_up[0,0]]
        titles = ['Input', 'Reconstruction', 'ALM', 'Latent Indices', 'NLL Map', 'Residual']
        for j in range(6):
            ax = axes[i, j]
            ax.imshow(imgs[j].cpu(), cmap='gray')
            ax.set_title(titles[j])
            ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{category_name}_grid.png")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--pixelcnn_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='inference_outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    ckpt_dir = Path(args.model_ckpt).parent
    config_path = ckpt_dir / "config.yaml"

    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = Args(**args_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitVQVAE.load_from_checkpoint(args.model_ckpt, args=model_args)
    model.to(device).eval()

    pixelcnn = load_pixelcnn_model(args.pixelcnn_ckpt, device=device)

    for category in CATEGORY_LIST:
        images, labels = load_category_samples(category)
        visualize_category(model, pixelcnn, images, labels, category, args.output_dir)

    print(f"All visualizations saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
