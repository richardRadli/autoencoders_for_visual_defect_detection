import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=cfg.get("input_channel"),
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[1],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[1],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[1],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            # Conv-6
            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[1],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            # Conv-7
            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            # Conv-8
            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            # Conv-9
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("latent_space_dimension"),
                      kernel_size=cfg.get("kernel_size")[2],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[0])
        )

        self.decoder = nn.Sequential(
            # DeConv-9
            nn.ConvTranspose2d(in_channels=cfg.get("latent_space_dimension"),
                               out_channels=cfg.get("flc")[0],
                               kernel_size=cfg.get("kernel_size")[2],
                               stride=cfg.get("stride")[0],
                               padding=cfg.get("padding")[0]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"),
                         inplace=True),
            # DeConv-8
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-7
            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0], padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-6
            nn.ConvTranspose2d(in_channels=cfg.get("flc")[2],
                               out_channels=cfg.get("flc")[1],
                               kernel_size=cfg.get("kernel_size")[1],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-5
            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-4
            nn.ConvTranspose2d(in_channels=cfg.get("flc")[1],
                               out_channels=cfg.get("flc")[0],
                               kernel_size=cfg.get("kernel_size")[1],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-3
            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-2
            nn.ConvTranspose2d(in_channels=cfg.get("flc")[0],
                               out_channels=cfg.get("flc")[0],
                               kernel_size=cfg.get("kernel_size")[1],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope"), inplace=True),
            # DeConv-1
            nn.ConvTranspose2d(in_channels=cfg.get("flc")[0],
                               out_channels=cfg.get("input_channel"),
                               kernel_size=cfg.get("kernel_size")[1],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
