import os
import cv2
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

# List of OCT image categories
CATEGORY_LIST = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Global lists to store anomaly scores and labels
all_scores = []
all_labels = []

def remove_margin(pil_img, threshold=240):
    """
    Remove white margin from a grayscale PIL image.
    Returns a cropped PIL image.
    """
    img_np = np.array(pil_img)
    mask = img_np < threshold
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return pil_img  # if image is all white, return original

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_np[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def load_category_samples(category, num_samples=3, base_dir='test_dataset'):
    """
    Load and preprocess sample images from a specific OCT category.

    Args:
        category: Category name (e.g., 'CNV', 'DME', etc.)
        num_samples: Number of images to load
        base_dir: Path to the base folder with images

    Returns:
        images: Tensor of shape (B, 1, 512, 512)
        labels: List of image filenames
    """
    transform = Compose([
        Resize((512, 512)),  # After cropping
        ToTensor()
    ])

    pattern = os.path.join(base_dir, f"{category}-*.jpeg")
    img_paths = sorted(glob(pattern))[:num_samples]

    images = []
    labels = []

    for path in img_paths:
        img = Image.open(path).convert("L")         # 1. Grayscale
        img = remove_margin(img, threshold=240)     # 2. Remove white margin
        img = transform(img)                        # 3. Resize + ToTensor
        images.append(img)
        labels.append(os.path.basename(path))

    return torch.stack(images), labels

def visualize_category(model, images, labels, category_name, save_dir):
    """
    Visualize model predictions for a category of images.
    
    For each image, generates:
    1. Input image
    2. Reconstructed image
    3. Alignment Loss Map (ALM)
    4. Residual map
    
    Also computes anomaly scores and AUROC metrics.
    
    Args:
        model: Trained VQVAE model
        images: Batch of input images
        labels: Image labels
        category_name: Name of the category
        save_dir: Directory to save visualizations
    """
    model.eval()
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))

    for i in range(3):
        # Run model inference
        x = images[i].unsqueeze(0).to(model.device)
        embedding_loss, x_hat, _, alm_map, encoding_indices, _, _ = model(x)

        # --- Residual Map Processing ---
        # Compute absolute difference between input and reconstruction
        residual_map = torch.abs(x - x_hat)
        res_np = residual_map[0, 0].detach().cpu().numpy()

        # Normalize for visualization (not used for scoring)
        res_norm = (res_np - res_np.min()) / (res_np.max() - res_np.min() + 1e-8)

        # Compute anomaly score using top-500 highest intensity pixels
        res_score = np.mean(np.sort(res_norm.flatten())[-500:])
        res_pred = 1 if res_score >= 0.5 else 0

        # --- Alignment Loss Map (ALM) Processing ---
        # Upsample ALM to match image size
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear')
        alm_np = alm_up[0, 0].detach().cpu().numpy()
        alm_np = cv2.GaussianBlur(alm_np, (5, 5), sigmaX=1)

        # Normalize for visualization (not used for scoring)
        alm_norm = (alm_np - alm_np.min()) / (alm_np.max() - alm_np.min() + 1e-8)

        # Compute anomaly score using top-500 highest intensity pixels
        alm_score = np.mean(np.sort(alm_norm.flatten())[-500:])
        alm_pred = 1 if alm_score >= 0.5 else 0

        # Ground truth label: 0 = Normal, 1 = Abnormal
        label = 0 if category_name == 'NORMAL' else 1

        # Format prediction info for plotting
        alm_str = (
            f"ALM\nScore={alm_score:.2f}"
        )
        res_str = (
            f"Residual\nScore={res_score:.2f}"
        )

        # Collect visual elements: input, reconstruction, ALM, residual
        imgs = [x[0, 0], x_hat[0, 0], alm_norm, res_norm]
        titles = ['Input', 'Reconstruction', alm_str, res_str]

        # Plot each image in the row
        for j in range(4):
            ax = axes[i, j]
            if j >= 2:
                ax.imshow(imgs[j], cmap='jet', vmin=0, vmax=1)  # Use color for ALM/Residual
            else:
                ax.imshow(imgs[j].detach().cpu(), cmap='gray')  # Use grayscale for input/reconstruction
            ax.set_title(titles[j], fontsize=10)
            ax.axis('off')

    # Save the figure for the category
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{category_name}_grid.png")
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Main inference function that:
    1. Loads trained model
    2. Processes each category of images
    3. Generates visualizations and metrics
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    # Setup paths and load model configuration
    model_ckpt = Path(args.model_dir)
    ckpt_dir = model_ckpt.parent
    config_path = Path(ckpt_dir) / "config.yaml"
    output_dir = os.path.join(ckpt_dir, 'inference')
    os.makedirs(output_dir, exist_ok=True)

    # Load model configuration
    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    # Create args object from config
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = Args(**args_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare model
    model = LitVQVAE.load_from_checkpoint(model_ckpt, args=model_args)
    model.to(device).eval()

    # Process each category
    for category in CATEGORY_LIST:
        images, labels = load_category_samples(category)
        visualize_category(model, images, labels, category, output_dir)

    print(f"All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
