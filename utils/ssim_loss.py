import torch
import torch.nn as nn
from kornia import metrics


class SSIMLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        max_val: float = 1.0,
        eps: float = 1e-12,
        reduction: str = "mean",
        padding: str = "same"
    ) -> None:
        """
        SSIM loss module.

        Args:
            window_size (int): The size of the sliding window for SSIM computation.
            max_val (float): The maximum possible pixel value (usually 1.0 for normalized images).
            eps (float): A small constant to avoid division by zero.
            reduction (str): Specifies the reduction to apply to the computed loss.
                Options: "mean", "sum", "none".
            padding (str): Specifies the padding for the SSIM computation. Options: "same", "valid".
        """

        super().__init__()

        self.window_size: int = window_size
        self.max_val: float = max_val
        self.eps: float = eps
        self.reduction: str = reduction
        self.padding: str = padding

    @staticmethod
    def ssim_loss(
            img1: torch.Tensor,
            img2: torch.Tensor,
            window_size: int,
            max_val: float = 1.0,
            eps: float = 1e-12,
            reduction: str = "mean",
            padding: str = "same",
    ) -> torch.Tensor:
        """
        Compute SSIM loss between two images.

        Args:
            img1 (torch.Tensor): The first image.
            img2 (torch.Tensor): The second image.
            window_size (int): The size of the sliding window for SSIM computation.
            max_val (float): The maximum possible pixel value (usually 1.0 for normalized images).
            eps (float): A small constant to avoid division by zero.
            reduction (str): Specifies the reduction to apply to the computed loss.
                Options: "mean", "sum", "none".
            padding (str): Specifies the padding for the SSIM computation. Options: "same", "valid".

        Returns:
            torch.Tensor: The computed SSIM loss.
        """

        ssim_map: torch.Tensor = metrics.ssim(img1, img2, window_size, max_val, eps, padding)
        loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction != "none":
            raise ValueError("Invalid reduction option. Choose from 'mean', 'sum', or 'none'.")

        return loss

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SSIM loss module.

        Args:
            img1 (torch.Tensor): The first image.
            img2 (torch.Tensor): The second image.

        Returns:
            torch.Tensor: The computed SSIM loss.
        """

        return self.ssim_loss(img1, img2, self.window_size, self.max_val, self.eps, self.reduction, self.padding)
