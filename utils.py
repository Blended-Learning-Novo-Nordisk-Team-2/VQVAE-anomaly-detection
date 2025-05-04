import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.io as io
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np


def load_cifar():
    """
    Load CIFAR-10 dataset with standard normalization.
    
    Returns:
        train: Training dataset
        val: Validation dataset
    """
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    """
    Load block dataset from numpy file.
    
    Returns:
        train: Training dataset
        val: Validation dataset
    """
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block(latent_path=None):
    """
    Load latent block dataset from saved latent representations.
    
    Args:
        latent_path: Path to latent representations file
        
    Returns:
        train: Training dataset
        val: Validation dataset
    """
    if latent_path is None:
        data_folder_path = os.getcwd()
        data_file_path = data_folder_path + \
        '/latent/latent_e_indices.npy'
    else:
        data_file_path = os.path.abspath(latent_path)

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):
    """
    Create DataLoader instances for training and validation data.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size for loading
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, latent_path):
    """
    Load dataset and create data loaders based on dataset type.
    
    Args:
        dataset: Dataset name ('CIFAR10', 'BLOCK', or 'LATENT_BLOCK')
        batch_size: Batch size for loading
        latent_path: Path to latent representations (for LATENT_BLOCK)
        
    Returns:
        training_data: Training dataset
        validation_data: Validation dataset
        training_loader: Training data loader
        validation_loader: Validation data loader
        x_train_var: Variance of training data
    """
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)
        
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block(latent_path)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    """
    Generate a readable timestamp string for file naming.
    
    Returns:
        str: Formatted timestamp string
    """
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_latents_from_model(model, dataloader, embedding_dim, save_path):
    """
    Extract and save latent representations from a trained model.
    
    Args:
        model: Trained VQVAE model
        dataloader: DataLoader for input data
        embedding_dim: Dimension of latent embeddings
        save_path: Path to save latent representations
    """
    model.eval()
    all_indices = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(next(model.parameters()).device)
            _, _, _, _, indices = model(x)
            all_indices.append(indices.cpu().numpy())
    all_latents = np.concatenate(all_indices, axis=0)  # [N, H, W]
    np.save(save_path, all_latents.astype(np.uint8))

if __name__ == "__main__":
    save_latents_from_model