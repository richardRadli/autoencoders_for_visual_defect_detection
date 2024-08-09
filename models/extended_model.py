import torch.nn as nn


class ExtendedAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(ExtendedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=cfg.get("input_channel"),
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[3],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[1],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[3],
                      out_channels=cfg.get("flc")[3],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[3],
                      out_channels=cfg.get("flc")[3],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[3],
                      out_channels=cfg.get("latent_space_dimension"),
                      kernel_size=cfg.get("kernel_size")[2],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[0])
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=cfg.get("latent_space_dimension"),
                               out_channels=cfg.get("flc")[3],
                               kernel_size=cfg.get("kernel_size")[2],
                               stride=cfg.get("stride")[0],
                               padding=cfg.get("padding")[0]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[3],
                      out_channels=cfg.get("flc")[3],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[3],
                      out_channels=cfg.get("flc")[3],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=cfg.get("flc")[3],
                               out_channels=cfg.get("flc")[2],
                               kernel_size=cfg.get("kernel_size")[0],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1],
                               output_padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[2],
                      out_channels=cfg.get("flc")[2],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=cfg.get("flc")[2],
                               out_channels=cfg.get("flc")[1],
                               kernel_size=cfg.get("kernel_size")[0],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1],
                               output_padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[1],
                      out_channels=cfg.get("flc")[1],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=cfg.get("flc")[1],
                               out_channels=cfg.get("flc")[0],
                               kernel_size=cfg.get("kernel_size")[0],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1],
                               output_padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=cfg.get("flc")[0],
                      out_channels=cfg.get("flc")[0],
                      kernel_size=cfg.get("kernel_size")[0],
                      stride=cfg.get("stride")[0],
                      padding=cfg.get("padding")[1]),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=cfg.get("flc")[0],
                               out_channels=cfg.get("input_channel"),
                               kernel_size=cfg.get("kernel_size")[1],
                               stride=cfg.get("stride")[1],
                               padding=cfg.get("padding")[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
