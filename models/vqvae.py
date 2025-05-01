
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, dropout=0.1, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim, dropout)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, dropout)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, return_extra=True):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, encoding_indices = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)
        alignment_loss_map = (z_e - z_q).pow(2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            print('alignment loss map:', alignment_loss_map.shape)
            print('encoding indices:', encoding_indices.shape)  # [B, H, W]
            assert False

        if return_extra:
            return embedding_loss, x_hat, perplexity, alignment_loss_map, encoding_indices

        return embedding_loss, x_hat, perplexity

class LitVQVAE(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = VQVAE(
            h_dim=args.n_hiddens,
            res_h_dim=args.n_residual_hiddens,
            n_res_layers=args.n_residual_layers,
            n_embeddings=args.n_embeddings,
            embedding_dim=args.embedding_dim,
            beta=args.beta,
            dropout=args.dropout
        )
        self.lr = args.learning_rate

    def forward(self, x):
        return self.model(x, return_extra=True)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device)
        embedding_loss, x_hat, perplexity, _, _ = self(x)
        recon_loss = torch.mean(torch.abs(x_hat - x))
        loss = recon_loss + embedding_loss

        self.log("train_total_loss", loss, on_step=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=True)
        self.log("train_embedding_loss", embedding_loss, on_step=True)
        self.log("train_perplexity", perplexity, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device)
        embedding_loss, x_hat, perplexity, _, _ = self(x)
        recon_loss = torch.mean(torch.abs(x_hat - x))
        loss = recon_loss + embedding_loss

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append({
            "val_total_loss": loss.detach(),
            "val_recon_loss": recon_loss.detach(),
            "val_perplexity": perplexity.detach()
        })

        return loss
    
    def on_validation_epoch_end(self):
        # 저장해둔 리스트에서 평균 계산
        if not hasattr(self, "val_outputs") or len(self.val_outputs) == 0:
            return

        total_loss = torch.stack([x["val_total_loss"] for x in self.val_outputs]).mean()
        recon_loss = torch.stack([x["val_recon_loss"] for x in self.val_outputs]).mean()
        perplexity = torch.stack([x["val_perplexity"] for x in self.val_outputs]).mean()

        self.log("val_total_loss", total_loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("val_perplexity", perplexity)

        # 초기화
        self.val_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)