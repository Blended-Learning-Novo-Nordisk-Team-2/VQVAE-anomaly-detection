import os
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

root = '../kermany2018'

class OCTDataset(Dataset):
    """
    OCTDataset for Normal -> Abnormal Anomaly Detection VAE.

    Now supports internal random train/val split from a single folder.
    """
    def __init__(self, root_dir, split='train', class_names=['NORMAL'], 
                 transform=None, val_split=0.1, subset='train', random_seed=42):
        """
        Args:
            root_dir (str): Root directory containing 'OCT2017' folder.
            split (str): Subfolder inside 'OCT2017' (typically 'train').
            class_names (list): List of class names to include.
            transform (callable, optional): Transform to apply to images.
            val_split (float): Fraction of data to reserve for validation.
            subset (str): 'train' or 'val' subset to return.
            random_seed (int): Random seed for reproducibility.
        """
        self.image_paths = []
        self.labels = []
        self.class_to_label = {'NORMAL': 0, 'ABNORMAL': 1}
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.subset = subset

        # Load all data first
        all_paths = []
        all_labels = []

        for class_name in class_names:
            folder = os.path.join(root_dir, 'OCT2017', split, class_name)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist!")
                continue
            for fname in os.listdir(folder):
                if fname.endswith('.jpeg') or fname.endswith('.jpg'):
                    all_paths.append(os.path.join(folder, fname))
                    label = 0 if class_name == 'NORMAL' else 1
                    all_labels.append(label)

        # Split into train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=val_split, random_state=random_seed, stratify=all_labels
        )

        if self.subset == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.subset == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        else:
            raise ValueError(f"Unknown subset type: {self.subset}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = self.transform(img)

        if idx == 0:
            if torch.isnan(img).any():
                print("Warning: NaN values found in image!")
            if torch.isinf(img).any():
                print("Warning: Inf values found in image!")

        return img, self.labels[idx]


def get_dataloaders(args):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = OCTDataset(
        root_dir=root,
        split='train',
        class_names=['NORMAL'],
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # 최종 출력 크기 512×512
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        val_split=0.1,
        subset='train'
    )

    val_dataset = OCTDataset(
        root_dir=root,
        split='train',
        class_names=['NORMAL'],
        transform=transform,
        val_split=0.1,
        subset='val'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
