import torch
import torch.nn as nn


class ExtendedAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(ExtendedAutoEncoder, self).__init__()

        # Extract configuration values
        input_channel = cfg.get("input_channel")
        flc = cfg.get("flc")
        kernel_size = cfg.get("kernel_size")
        stride = cfg.get("stride")
        padding = cfg.get("padding")
        alpha_slope = cfg.get("alpha_slope")
        latent_space_dimension = cfg.get("latent_space_dimension")

        # Encoder layers
        self.encoder = nn.Sequential(
            self._conv_block(input_channel, flc[0], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[1], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[2], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[3], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            nn.Conv2d(flc[3], latent_space_dimension, kernel_size[1], stride[0], padding[0])
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            self._deconv_block(latent_space_dimension, flc[3], kernel_size[1], stride[0], padding[0], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[3], flc[2], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[2], flc[1], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[1], flc[0], kernel_size[0], stride[1], padding[1], alpha_slope),
            nn.ConvTranspose2d(flc[0], input_channel, kernel_size[1], stride=1, padding=padding[1]),
            nn.Sigmoid()
        )

    @staticmethod
    def _conv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, alpha_slope: float
    ) -> nn.Sequential:
        """
        Helper method to create a convolutional block with LeakyReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding applied to the input.
            alpha_slope (float): Slope for the LeakyReLU activation function.

        Returns:
            nn.Sequential: A sequential container with convolution and LeakyReLU layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=alpha_slope, inplace=True)
        )

    @staticmethod
    def _deconv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, alpha_slope: float
    ) -> nn.Sequential:
        """
        Helper method to create a transposed convolutional block with LeakyReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the transposed convolutional kernel.
            stride (int): Stride of the transposed convolution.
            padding (int): Padding applied to the input.
            alpha_slope (float): Slope for the LeakyReLU activation function.

        Returns:
            nn.Sequential: A sequential container with transposed convolution and LeakyReLU layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=alpha_slope, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
