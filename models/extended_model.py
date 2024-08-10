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

        self.encoder = nn.Sequential(
            self._conv_block(input_channel, flc[0], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[1], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[2], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[3], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            nn.Conv2d(flc[3], latent_space_dimension, kernel_size[2], stride[0], padding[0]),
        )

        self.decoder = nn.Sequential(
            self._deconv_block(latent_space_dimension, flc[3], kernel_size[2], stride[0], padding[0], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[3], flc[3], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[3], flc[2], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[2], flc[2], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[2], flc[1], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[1], flc[1], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._deconv_block(flc[1], flc[0], kernel_size[0], stride[1], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._conv_block(flc[0], flc[0], kernel_size[0], stride[0], padding[1], alpha_slope),
            self._last_deconv_block(flc[0], input_channel, kernel_size[1], stride[1], padding[1])
        )

    @staticmethod
    def _conv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, alpha_slope: float
    ) -> nn.Sequential:
        """
        Helper method to create a convolutional block with LeakyReLU activation.

        Args:
            in_channels (int): The number of input channels to the convolutional layer.
            out_channels (int): The number of output channels produced by the convolutional layer.
            kernel_size (int): The size of the convolutional kernel (both height and width).
            stride (int): The stride of the convolution operation.
            padding (int): The amount of padding added to both sides of the input.
            alpha_slope (float): The slope of the LeakyReLU activation function for negative inputs.

        Returns:
            nn.Sequential: A PyTorch `nn.Sequential` module containing the convolutional layer and
                            the LeakyReLU activation function.
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

        Helper method to create a convolutional block with LeakyReLU activation.

        Args:
            in_channels (int): The number of input channels to the convolutional layer.
            out_channels (int): The number of output channels produced by the convolutional layer.
            kernel_size (int): The size of the convolutional kernel (both height and width).
            stride (int): The stride of the convolution operation.
            padding (int): The amount of padding added to both sides of the input.
            alpha_slope (float): The slope of the LeakyReLU activation function for negative inputs.

        Returns:
            nn.Sequential: A PyTorch `nn.Sequential` module containing the convolutional layer and
                            the LeakyReLU activation function.
        """

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=padding),
            nn.LeakyReLU(negative_slope=alpha_slope, inplace=True)
        )

    @staticmethod
    def _last_deconv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> nn.Sequential:
        """
        Helper method to create a transposed convolutional block with Sigmoid activation.

        Helper method to create a convolutional block with LeakyReLU activation.

        Args:
            in_channels (int): The number of input channels to the convolutional layer.
            out_channels (int): The number of output channels produced by the convolutional layer.
            kernel_size (int): The size of the convolutional kernel (both height and width).
            stride (int): The stride of the convolution operation.
            padding (int): The amount of padding added to both sides of the input.

        Returns:
            nn.Sequential: A PyTorch `nn.Sequential` module containing the convolutional layer and
                            the Sigmoid activation function.
        """

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
