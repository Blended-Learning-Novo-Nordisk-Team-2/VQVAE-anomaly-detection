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
    OCT (Optical Coherence Tomography) Dataset for Anomaly Detection.
    
    Features:
    - Supports internal train/validation split
    - Handles both normal and abnormal classes
    - Includes data augmentation for training
    - Converts images to grayscale
    
    Parameters:
    - root_dir: Path to dataset root directory
    - split: Data split ('train' or 'val')
    - class_names: List of classes to include
    - transform: Image transformations
    - val_split: Validation set ratio
    - subset: Which subset to return ('train' or 'val')
    - random_seed: Random seed for reproducibility
    """
    def __init__(self, root_dir, split='train', class_names=['NORMAL'], 
                 transform=None, val_split=0.1, subset='train', random_seed=42):
        self.image_paths = []
        self.labels = []
        self.class_to_label = {'NORMAL': 0, 'ABNORMAL': 1}
        
        # Default transformations if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.subset = subset

        # Load all data first
        all_paths = []
        all_labels = []

        # Collect all image paths and labels
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

        # Split data into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=val_split, random_state=random_seed, stratify=all_labels
        )

        # Select appropriate subset
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
        # Load and preprocess image
        img = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        img = self.transform(img)

        # Check for numerical issues in first image
        if idx == 0:
            if torch.isnan(img).any():
                print("Warning: NaN values found in image!")
            if torch.isinf(img).any():
                print("Warning: Inf values found in image!")

        return img, self.labels[idx]


def get_dataloaders(args):
    """
    Create training and validation dataloaders with appropriate transformations.
    
    Training transformations include:
    - Random horizontal flips
    - Random rotations
    - Random resized crops
    - Gaussian blur for augmentation
    
    Validation transformations only include:
    - Resize to 512x512
    - Normalization
    """
    # Validation transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Training dataset with augmentations
    train_dataset = OCTDataset(
        root_dir=root,
        split='train',
        class_names=['NORMAL'],
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flips
            transforms.RandomRotation(degrees=15),  # Random rotations
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # Random crops
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        val_split=0.1,
        subset='train'
    )

    # Validation dataset without augmentations
    val_dataset = OCTDataset(
        root_dir=root,
        split='train',
        class_names=['NORMAL'],
        transform=transform,
        val_split=0.1,
        subset='val'
    )

    # Create dataloaders with specified batch size and workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
