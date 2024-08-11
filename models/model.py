import torch
import torch.nn as nn
import torch.nn.init as init


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()

        # Extract configuration values
        self.input_channel = cfg.get("input_channel")
        self.flc = cfg.get("flc")
        self.kernel_size = cfg.get("kernel_size")
        self.stride = cfg.get("stride")
        self.padding = cfg.get("padding")
        self.alpha_slope = cfg.get("alpha_slope")
        self.latent_space_dimension = cfg.get("latent_space_dimension")

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
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, alpha_slope: float = None
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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=alpha_slope, inplace=True) if alpha_slope else nn.Sigmoid()
        )

    @staticmethod
    def _deconv_block_extended(
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

    def _initialize_weights(self):
        """
        Initialize weights for the model using Kaiming initialization for convolutional layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

