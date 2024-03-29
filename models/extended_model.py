import torch.nn as nn


class ExtendedAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(ExtendedAutoEncoder, self).__init__()

        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.cfg.get("input_channel"),
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[1],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[0], 
                      out_channels=self.cfg.get("flc")[0], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[0], 
                      out_channels=self.cfg.get("flc")[0], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[0], 
                      out_channels=self.cfg.get("flc")[1], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[1], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[1], 
                      out_channels=self.cfg.get("flc")[1], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[1], 
                      out_channels=self.cfg.get("flc")[1], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[1], 
                      out_channels=self.cfg.get("flc")[2], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[1], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[2], 
                      out_channels=self.cfg.get("flc")[2], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[2], 
                      out_channels=self.cfg.get("flc")[2], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[2], 
                      out_channels=self.cfg.get("flc")[3], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[1], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[3], 
                      out_channels=self.cfg.get("flc")[3], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[3], 
                      out_channels=self.cfg.get("flc")[3], 
                      kernel_size=self.cfg.get("kernel_size")[0], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[3], 
                      out_channels=self.cfg.get("latent_space_dimension"), 
                      kernel_size=self.cfg.get("kernel_size")[2], 
                      stride=self.cfg.get("stride")[0], 
                      padding=self.cfg.get("padding")[0])
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.cfg.get("latent_space_dimension"),
                               out_channels=self.cfg.get("flc")[3],
                               kernel_size=self.cfg.get("kernel_size")[2],
                               stride=self.cfg.get("stride")[0],
                               padding=self.cfg.get("padding")[0]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[3],
                      out_channels=self.cfg.get("flc")[3],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[3],
                      out_channels=self.cfg.get("flc")[3],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[3],
                               out_channels=self.cfg.get("flc")[2],
                               kernel_size=self.cfg.get("kernel_size")[0],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1],
                               output_padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[2],
                      out_channels=self.cfg.get("flc")[2],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[2],
                      out_channels=self.cfg.get("flc")[2],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[2],
                               out_channels=self.cfg.get("flc")[1],
                               kernel_size=self.cfg.get("kernel_size")[0],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1],
                               output_padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[1],
                      out_channels=self.cfg.get("flc")[1],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[1],
                               out_channels=self.cfg.get("flc")[0],
                               kernel_size=self.cfg.get("kernel_size")[0],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1],
                               output_padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.LeakyReLU(negative_slope=self.cfg.get("alpha_slope")),

            nn.Conv2d(in_channels=self.cfg.get("flc")[0],
                      out_channels=self.cfg.get("flc")[0],
                      kernel_size=self.cfg.get("kernel_size")[0],
                      stride=self.cfg.get("stride")[0],
                      padding=self.cfg.get("padding")[1]),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=self.cfg.get("flc")[0],
                               out_channels=self.cfg.get("input_channel"),
                               kernel_size=self.cfg.get("kernel_size")[1],
                               stride=self.cfg.get("stride")[1],
                               padding=self.cfg.get("padding")[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
