from torchvision.models import vgg16
import torch.nn.functional as F
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15), scale_factor=1.0):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.selected_layers = layers
        self.scale_factor = scale_factor

        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, x_hat):
        if x.min() < 0:
            x = (x + 1) / 2
            x_hat = (x_hat + 1) / 2

        if self.scale_factor != 1.0:
            size = [int(x.size(2) * self.scale_factor), int(x.size(3) * self.scale_factor)]
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            x_hat = F.interpolate(x_hat, size=size, mode='bilinear', align_corners=False)

        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            x_hat = layer(x_hat)
            if i in self.selected_layers:
                loss += F.l1_loss(x, x_hat)
        return loss
    
class MultiScaleResidualLoss(nn.Module):
    def __init__(self, scales=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales

    def forward(self, x, x_hat):
        total_loss = 0.0
        for scale in self.scales:
            if scale == 1.0:
                x_scaled = x
                x_hat_scaled = x_hat
            else:
                size = [int(x.size(2) * scale), int(x.size(3) * scale)]
                x_scaled = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                x_hat_scaled = F.interpolate(x_hat, size=size, mode='bilinear', align_corners=False)
            total_loss += F.l1_loss(x_scaled, x_hat_scaled)
        return total_loss
    
class LatentConsistencyLoss(nn.Module):
    def __init__(self, mode='mse'):
        super().__init__()
        self.mode = mode

    def forward(self, z_e, z_q):
        if self.mode == 'mse':
            return F.mse_loss(z_e, z_q)
        elif self.mode == 'l1':
            return F.l1_loss(z_e, z_q)
        elif self.mode == 'cosine':
            return 1 - F.cosine_similarity(z_e.flatten(1), z_q.flatten(1)).mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")