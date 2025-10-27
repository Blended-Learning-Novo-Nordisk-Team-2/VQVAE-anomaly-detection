import os
import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.vqvae import LitVQVAE
from vit_mlp_module.vit_mlp import VITMLP
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from glob import glob
from pathlib import Path
import yaml

CATEGORY_LIST = ["CNV", "DME", "DRUSEN", "NORMAL"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_checkpoint', type=str, required=True)
    parser.add_argument('--vqvae_checkpoint', type=str, required=True)
    parser.add_argument('--mlp_checkpoint', type=str, default=None)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='inference_combined')
    return parser.parse_args()

def remove_margin(pil_img, threshold=240):
    img_np = np.array(pil_img)
    mask = img_np < threshold
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_np[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def load_category_samples(category, num_samples=None, base_dir='test_dataset'):
    transform = Compose([
        Resize((512, 512)),
        ToTensor()
    ])
    pattern = os.path.join(base_dir, f"{category}-*.jpeg")
    img_paths = sorted(glob(pattern))
    if num_samples is not None:
        img_paths = img_paths[:num_samples]
    images = []
    labels = []
    for path in img_paths:
        img = Image.open(path).convert("L")
        img = remove_margin(img, threshold=240)
        img = transform(img)
        images.append(img)
        labels.append(os.path.basename(path))
    return torch.stack(images), labels

def load_separate_models(vit_checkpoint, vqvae_checkpoint, mlp_checkpoint=None, device='cuda'):
    class Args:
        def __init__(self):
            self.vit_checkpoint = vit_checkpoint
            self.vqvae_checkpoint = vqvae_checkpoint
            self.device = device
            self.out_size = 64
            self.n_embeddings = 32
    args = Args()
    model = VITMLP(args)
    if mlp_checkpoint:
        mlp_state = torch.load(mlp_checkpoint, map_location=device)
        state_dict = mlp_state.get('model_state', mlp_state.get('state_dict', mlp_state))
        mlp_state_dict = {
            k.replace('mlp_prior.', ''): v
            for k, v in state_dict.items()
            if k.startswith('mlp_prior')
        }
        if mlp_state_dict:
            model.mlp_prior.load_state_dict(mlp_state_dict)
    model.eval()
    return model

def visualize_category_combined(vqvae_model, vitmlp_model, images, labels, category_name, save_dir):
    vqvae_model.eval()
    vitmlp_model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(images)):
        x = images[i].unsqueeze(0).to(vqvae_model.device)

        # VQVAE forward
        _, x_hat, _, alm_map, _, _, _ = vqvae_model(x)
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear')
        alm_np = alm_up[0, 0].detach().cpu().numpy()
        alm_np = cv2.GaussianBlur(alm_np, (5, 5), sigmaX=1)
        alm_norm = (alm_np - alm_np.min()) / (alm_np.max() - alm_np.min() + 1e-8)

        # NLL map (using VQVAE's true latent indices)
        with torch.no_grad():
            gt_indices = vitmlp_model.get_vqvae_latents(x)  # [B, H, W]
            vit_input = x.repeat(1, 3, 1, 1)
            logits, probs = vitmlp_model(vit_input, return_probs=True)  # [B, n_embed, H, W]
            selected_probs = torch.gather(probs, 1, gt_indices.unsqueeze(1))  # [B, 1, H, W]
            nll_map = -torch.log(selected_probs + 1e-6)  # [B, 1, H, W]
            nll_up = F.interpolate(nll_map, size=(512, 512), mode='bilinear', align_corners=False)
            
            # VIT 모델을 위한 입력 이미지 리사이즈
            vit_input_resized = F.interpolate(vit_input, size=(256, 256), mode='bilinear', align_corners=False)
            # VIT 모델을 직접 사용하여 예측
            vit_pred = vitmlp_model.vit(vit_input_resized)  # [B, num_classes]
            pred = torch.argmax(vit_pred, dim=1)  # [B]
            pred_label = "Normal" if pred.item() == 0 else "Abnormal"
            true_label = "Normal" if category_name == "NORMAL" else "Abnormal"

        def to_numpy(tensor, is_nll=False, invert=False):
            t = tensor.detach().squeeze().cpu().numpy()
            if is_nll:
                t = np.clip(t, 0, np.percentile(t, 98))
            norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
            return 1 - norm if invert else norm

        # 원본 이미지와 히트맵 오버레이를 위한 함수
        def overlay_heatmap(img, heatmap, alpha=0.6, is_nll=False):
            img_np = to_numpy(img)
            heatmap_np = to_numpy(heatmap, invert=is_nll)
            heatmap_colored = plt.cm.jet(heatmap_np)[:, :, :3]  # RGB로 변환
            overlay = alpha * heatmap_colored + (1 - alpha) * img_np[:, :, np.newaxis]
            return np.clip(overlay, 0, 1)

        # 원본 이미지
        input_img = to_numpy(x)
        
        # ALM과 NLL 오버레이
        alm_overlay = overlay_heatmap(x, torch.from_numpy(alm_norm).unsqueeze(0).unsqueeze(0), is_nll=False)
        nll_overlay = overlay_heatmap(x, nll_up, is_nll=True)

        imgs = [input_img, to_numpy(x_hat), alm_overlay, nll_overlay]
        titles = ['Input', 'Reconstruction', 'ALM Overlay', 'NLL Overlay']
        
        # 전체 figure 크기 조정 (아래 텍스트를 위한 공간 확보)
        fig = plt.figure(figsize=(24, 7))
        
        # 이미지 그리드를 위한 subplot 생성
        gs = fig.add_gridspec(2, 4, height_ratios=[4, 1])
        
        # 이미지 표시
        for j in range(4):
            ax = fig.add_subplot(gs[0, j])
            if j < 2:  # Input과 Reconstruction은 그레이스케일
                ax.imshow(imgs[j], cmap='gray')
            else:  # ALM과 NLL 오버레이는 컬러
                ax.imshow(imgs[j])
            ax.set_title(titles[j], fontsize=12)
            ax.axis('off')
        
        # 결과 텍스트를 위한 subplot
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')
        result_text = f"True: {true_label} | Pred: {pred_label}"
        ax_text.text(0.5, 0.5, result_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    fontweight='bold')

        fname = labels[i].replace('.jpeg', '_combined.png')
        save_path = os.path.join(save_dir, f"{category_name}_{fname}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vqvae_ckpt_path = Path(args.vqvae_checkpoint)
    config_path = vqvae_ckpt_path.parent / "config.yaml"
    with open(config_path, "r") as f:
        args_dict = yaml.safe_load(f)

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = Args(**args_dict)
    vqvae_model = LitVQVAE.load_from_checkpoint(args.vqvae_checkpoint, args=model_args)
    vqvae_model.to(device).eval()

    vit_model = load_separate_models(
        vit_checkpoint=args.vit_checkpoint,
        vqvae_checkpoint=args.vqvae_checkpoint,
        mlp_checkpoint=args.mlp_checkpoint,
        device=device
    )
    vit_model.to(device).eval()

    for category in CATEGORY_LIST:
        images, labels = load_category_samples(category, base_dir=args.test_dir)  # num_samples 제거
        visualize_category_combined(
            vqvae_model=vqvae_model,
            vitmlp_model=vit_model,
            images=images,
            labels=labels,
            category_name=category,
            save_dir=args.save_dir
        )
    print(f"All visualizations saved to {args.save_dir}")

if __name__ == '__main__':
    main()