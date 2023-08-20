import torch.nn as nn
from pytorch_ssim import ssim


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    @staticmethod
    def forward(predicted, target):
        return 1 - ssim(predicted, target)
