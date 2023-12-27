import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.encoder = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels=self.cfg.get("input_channel"),
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[1],  # 4
                      stride=self.cfg.get("stride")[1],  # 2
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 2
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[1],  # 4
                      stride=self.cfg.get("stride")[1],  # 2
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 3
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],  # 3
                      stride=self.cfg.get("stride")[0],  # 1
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 4
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[1],  # 4
                      stride=self.cfg.get("stride")[1],  # 2
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 5
            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],  # 3
                      stride=self.cfg.get("stride")[0],  # 1
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 6
            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[2],
                      kernel_size=self.cfg.get("kernel_size")[1],  # 4
                      stride=self.cfg.get("stride")[1],  # 2
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 7
            nn.Conv2d(in_channels=self.cfg.get("flc")[2],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],  # 3
                      stride=self.cfg.get("stride")[0],  # 1
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 8
            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],  # 3
                      stride=self.cfg.get("stride")[0],  # 1
                      padding=self.cfg.get("padding")[1]),  # 1
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 9
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("latent_space_dimension"),
                      kernel_size=self.cfg.get("kernel_size")[2],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[0])
        )

        self.decoder = nn.Sequential(
            # Conv 8
            nn.ConvTranspose2d(in_channels=self.cfg.get("latent_space_dimension"),
                               out_channels=self.cfg.get("flc")[0],
                               kernel_size=self.cfg.get("kernel_size")[2],
                               stride=self.cfg.get("stride")[0],
                               padding=self.cfg.get("padding")[0]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # # Conv 7
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 6
            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[2],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 5
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[2],
                               out_channels=self.cfg.get("flc")[1],
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # # Conv 4
            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 3
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[1],
                               out_channels=self.cfg.get("flc")[0],
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 2
            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            # Conv 1
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[0],
                               out_channels=self.cfg.get("flc")[0],
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[0],
                               out_channels=self.cfg.get("input_channel"),
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
