import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Dataset for processing block images.
    
    Features:
    - Loads and resizes images to 32x32
    - Supports RGB images (3 channels)
    - Splits data into train/validation sets
    - Applies optional transformations
    
    Parameters:
    - file_path: Path to numpy file containing image data
    - train: Whether to use training or validation split
    - transform: Optional image transformations
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        # Load data from numpy file
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        
        # Resize all images to 32x32 using cubic interpolation
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        # Split data into train/validation (90/10 split)
        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        # Get image and apply transformations if specified
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0  # All images are labeled as 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Dataset for processing latent block representations.
    
    Features:
    - Loads pre-computed latent representations
    - Fixed train/validation split (500 samples for validation)
    - Supports optional transformations
    
    Parameters:
    - file_path: Path to numpy file containing latent data
    - train: Whether to use training or validation split
    - transform: Optional transformations
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading latent block data')
        # Load latent representations from numpy file
        data = np.load(file_path, allow_pickle=True)
        print('Done loading latent block data')
        
        # Split data into train/validation (500 samples for validation)
        self.data = data[:-500] if train else data[-500:]
        self.transform = transform

    def __getitem__(self, index):
        # Get latent representation and apply transformations if specified
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0  # All samples are labeled as 0
        return img, label

    def __len__(self):
        return len(self.data)
