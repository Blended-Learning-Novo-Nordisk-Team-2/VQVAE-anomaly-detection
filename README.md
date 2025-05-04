# VQVAE for OCT Image Anomaly Detection

This repository implements a Vector Quantized Variational AutoEncoder (VQVAE) for anomaly detection in Optical Coherence Tomography (OCT) images. The model is trained on normal OCT images and can detect anomalies by measuring reconstruction error and alignment loss, with additional prior-based NLL implementation coming soon.

The implementation is based on two main sources:
1. The core VQVAE architecture and training pipeline is adapted from [MishaLaskin's PyTorch implementation](https://github.com/MishaLaskin/vqvae.git)
2. The model structure and anomaly detection approach follows the methodology described in the paper ["Anomaly Detection in Optical Coherence Tomography Angiography (OCTA) with a Vector-Quantized Variational Auto-Encoder (VQ-VAE)"](https://www.mdpi.com/2306-5354/11/7/682) (2024)


## Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd vqvae
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Project Structure
```
project_root/
├── kermany2018/              # OCT dataset (should be in parent directory)
│   └── OCT2017/
│       ├── train/
│       │   ├── NORMAL/
│       │   ├── CNV/
│       │   ├── DME/
│       │   └── DRUSEN/
│       └── test/
├── vqvae/                    # Project directory
│   ├── models/               # Model architecture files
│   ├── datasets/             # Dataset handling code
│   ├── saved_models/         # Training outputs
│   │   └── YYYY-MM-DD_HH-MM-SS/
│   │       ├── best-*.ckpt   # Best model checkpoint
│   │       ├── config.yaml   # Training configuration
│   │       └── latent/       # Latent representations
│   ├── test_dataset/         # Test images for inference
│   ├── train.py             # Training script
│   ├── simple_inference.py  # Inference script
│   └── utils.py             # Utility functions
```

Note: If your `kermany2018` dataset is located in a different directory, modify the `root` variable in `datasets/dataset.py`:
```python
root = '../path/to/your/kermany2018'
```

## Usage

### Training

The training script supports various hyperparameters through command-line arguments. Here's the basic usage:

```bash
python train.py --batch_size 16 --epochs 100 --n_hiddens 256 --embedding_dim 256 --n_embeddings 128 --beta 0.25 --learning_rate 1e-4
```

Required arguments:
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--n_hiddens`: Hidden dimension size
- `--embedding_dim`: Dimension of codebook vectors
- `--n_embeddings`: Size of codebook
- `--beta`: Commitment loss coefficient
- `--learning_rate`: Learning rate for optimization

Optional arguments:
- `--n_residual_hiddens`: Hidden dimension in residual blocks (default: 64)
- `--n_residual_layers`: Number of residual layers (default: 4)
- `--dropout`: Dropout rate (default: 0.1)
- `--patience`: Early stopping patience (default: 10)
- `--min_delta`: Minimum change for early stopping (default: 1e-4)
- `--device`: Training device (default: "gpu")
- `--save`: Whether to save model outputs (default: True)

During training:
1. A new directory is created in `saved_models/` with the current timestamp
2. Model checkpoints are saved in this directory
3. Latent representations are saved in the `latent/` subdirectory
4. Training configuration is saved as `config.yaml`

### Inference and Visualization

After training, you can run inference and generate visualizations using:

```bash
python inference.py --model_dir path/to/saved_models/YYYY-MM-DD_HH-MM-SS/best-*.ckpt
```

Required arguments:
- `--model_dir`: Path to the model checkpoint file

The script will:
1. Load the trained model
2. Process images from each category (CNV, DME, DRUSEN, NORMAL)
3. Generate visualizations including:
   - Input images
   - Reconstructed images
   - Alignment Loss Maps (ALM)
   - Residual maps
4. Save visualizations in the `inference/` directory within the model's directory
5. Compute and display anomaly scores and AUROC metrics

## Model Architecture

The VQVAE consists of:
- Encoder: Converts input images to continuous latent representations
- Vector Quantizer: Discretizes continuous latent vectors using a codebook
- Decoder: Reconstructs images from quantized latent vectors

The model uses residual blocks for better feature extraction and reconstruction.

## Hyperparameter Tuning

You can use the provided `sweep.yaml` for hyperparameter optimization with Weights & Biases. The sweep configuration includes:
- Learning rate: [1e-3, 1e-4, 1e-5]
- Beta: [0.1, 0.25, 0.5]
- Embedding dimension: [64, 128]
- Number of embeddings: [32, 64]
- Hidden dimensions: [128, 256]
- Dropout: [0.1, 0.2]

## License

Not Yet
