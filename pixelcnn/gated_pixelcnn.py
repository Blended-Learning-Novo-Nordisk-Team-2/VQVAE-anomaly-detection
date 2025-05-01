import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os 
import sys

"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd() + '/pixelcnn')

from pixelcnn.models import GatedPixelCNN
import utils

train_loader = None
test_loader = None
model = None
opt = None
criterion = None

"""
train, test, and log
"""

def train():
    global train_loader, model, opt, criterion, args  # ← 반드시 함수 가장 첫 줄
    print(f"[TRAIN] Starting training loop with {len(train_loader)} batches")

    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()
        
        if args.dataset == 'LATENT_BLOCK':
            x = (x[:, 0]).unsqueeze(-1).long().cuda()
        else:
            x = (x[:, 0] * (args.n_embeddings - 1)).unsqueeze(-1).long().cuda()
        label = label.cuda()
    
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                args.log_interval * batch_idx / len(train_loader),
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))


def test():
    global test_loader, model, criterion, args
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            if args.dataset == 'LATENT_BLOCK':
                x = (x[:, 0]).unsqueeze(-1).long().cuda()
            else:
                x = (x[:, 0] * (args.n_embeddings-1)).unsqueeze(-1).long().cuda()
            label = label.cuda()

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)

def generate_samples(epoch):
    global test_loader, model, criterion, args
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().cuda()
    x_tilde = model.generate(label, shape=(args.img_dim, args.img_dim), batch_size=100)
    print(x_tilde[0])


if __name__ == "__main__":
    """
    Hyperparameters
    """
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-gen_samples", action="store_true")

    parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
        help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_dim", type=int, default=64)
    parser.add_argument("--input_dim", type=int, default=1,
        help='1 for grayscale 3 for rgb')
    parser.add_argument("--n_embeddings", type=int, default=128,
        help='number of embeddings from VQ VAE')
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='pixelcnn/checkpoints')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    data loaders
    """
    try:
        if args.dataset == 'LATENT_BLOCK':
            print("[DEBUG] Loading LATENT_BLOCK data...")
            _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders(
                'LATENT_BLOCK', args.batch_size, args.data_path
            )
            print(f"[INFO] Save directory is: {args.save_dir}")
            print(f"[INFO] Save flag is: {args.save}")
            print(f"[INFO] Dataset is: {args.dataset}")
            print(f"[INFO] Train loader has {len(train_loader)} batches")
            print("[DEBUG] LATENT_BLOCK data loaded.")
        else:
            raise ValueError("Only LATENT_BLOCK supported for now.")
    except Exception as e:
        print("[ERROR] Failed to load data loaders:")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    model = GatedPixelCNN(args.n_embeddings, args.img_dim**2, args.n_layers).to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    BEST_LOSS = 999
    LAST_SAVED = -1

    print(f"[DEBUG] Number of training batches: {len(train_loader)}")
    print(f"[DEBUG] Data path: {args.data_path}")
    print(f"[DEBUG] args.epochs = {args.epochs}")
    print(f"[DEBUG] Starting epoch loop...")

    for epoch in range(1, args.epochs):
        print("\nEpoch {}:".format(epoch))
        train()
        cur_loss = test()

        if args.save or cur_loss <= BEST_LOSS:
            BEST_LOSS = cur_loss
            LAST_SAVED = epoch

            print("Saving model!")
            pixelcnn_save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({'model_state_dict': model.state_dict()}, pixelcnn_save_path)
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

        if args.gen_samples:
            generate_samples(epoch)
