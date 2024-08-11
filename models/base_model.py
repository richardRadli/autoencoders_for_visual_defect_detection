import torch
import torch.nn as nn

from models.model import AutoEncoder


class BaseAutoEncoder(AutoEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Encoder layers
        self.encoder = nn.Sequential(
            self._conv_block(
                self.input_channel, self.flc[0], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[0], self.flc[0], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[0], self.flc[0], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[0], self.flc[1], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[1], self.flc[1], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[1], self.flc[2], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[2], self.flc[1], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[1], self.flc[0], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            nn.Conv2d(
                self.flc[0], self.latent_space_dimension, self.kernel_size[2], self.stride[0], self.padding[0]
            )
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            self._deconv_block(
                self.latent_space_dimension, self.flc[0], self.kernel_size[2], self.stride[0], self.padding[0], self.alpha_slope
            ),
            self._conv_block(
                self.flc[0], self.flc[1], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[1], self.flc[2], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._deconv_block(
                self.flc[2], self.flc[1], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[1], self.flc[1], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._deconv_block(
                self.flc[1], self.flc[0], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._conv_block(
                self.flc[0], self.flc[0], self.kernel_size[0], self.stride[0], self.padding[1], self.alpha_slope
            ),
            self._deconv_block(
                self.flc[0], self.flc[0], self.kernel_size[1], self.stride[1], self.padding[1], self.alpha_slope
            ),
            self._deconv_block(
                self.flc[0], self.input_channel, self.kernel_size[1], self.stride[1], self.padding[1]
            )
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): A tensor containing the input data.

        Returns:
            torch.Tensor: A tensor containing the output data.
        """

        x = self.encoder(x)
        x = self.decoder(x)
        return x
