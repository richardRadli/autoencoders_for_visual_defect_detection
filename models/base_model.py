import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super(AutoEncoder, self).__init__()

        self.cfg = cfg

        # E N C O D E R
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.cfg.get("input_channel"),
                      out_channels=self.cfg.get("flc"),
                      kernel_size=self.cfg.get("kernel_size")[1],
                      stride=self.cfg.get("stride")[1],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("flc"),
                      kernel_size=self.cfg.get("kernel_size")[1],
                      stride=self.cfg.get("stride")[1],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("flc"),
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("flc") * 2,
                      kernel_size=self.cfg.get("kernel_size")[1],
                      stride=self.cfg.get("stride")[1],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 2,
                      out_channels=self.cfg.get("flc") * 2,
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 2,
                      out_channels=self.cfg.get("flc") * 4,
                      kernel_size=self.cfg.get("kernel_size")[1],
                      stride=self.cfg.get("stride")[1],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 4,
                      out_channels=self.cfg.get("flc") * 2,
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 2,
                      out_channels=self.cfg.get("flc"),
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("latent_space_dimension"),
                      kernel_size=self.cfg.get("kernel_size")[2],
                      stride=self.cfg.get("stride")[0])
        )

        # D E C O D E R
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.cfg.get("latent_space_dimension"),
                               out_channels=self.cfg.get("flc"),
                               kernel_size=self.cfg.get("kernel_size")[2],
                               stride=self.cfg.get("stride")[0]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("flc") * 2,
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 2,
                      out_channels=self.cfg.get("flc") * 4,
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc") * 4,
                               out_channels=self.cfg.get("flc") * 2,
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc") * 2,
                      out_channels=self.cfg.get("flc") * 2,
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc") * 2,
                               out_channels=self.cfg.get("flc"),
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.Conv2d(in_channels=self.cfg.get("flc"),
                      out_channels=self.cfg.get("flc"),
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),
            nn.ConvTranspose2d(in_channels=self.cfg.get("flc"),
                               out_channels=self.cfg.get("input_channel"),
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = nn.functional.interpolate(input=decoded,
                                            size=self.cfg.get("img_size"),
                                            mode='bilinear',
                                            align_corners=False)
        return decoded
