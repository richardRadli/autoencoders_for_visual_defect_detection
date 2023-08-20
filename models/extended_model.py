import torch.nn as nn


class ExtendedAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(ExtendedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.input_channel, cfg.flc, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc, cfg.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc, cfg.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(cfg.flc, cfg.flc * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 2, cfg.flc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 2, cfg.flc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(cfg.flc * 2, cfg.flc * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 4, cfg.flc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 4, cfg.flc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(cfg.flc * 4, cfg.flc * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 8, cfg.flc * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 8, cfg.flc * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(cfg.flc * 8, cfg.z_dim, kernel_size=8, stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(cfg.z_dim, cfg.flc * 8, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 8, cfg.flc * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 8, cfg.flc * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(cfg.flc * 8, cfg.flc * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 4, cfg.flc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 4, cfg.flc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(cfg.flc * 4, cfg.flc * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 2, cfg.flc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc * 2, cfg.flc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(cfg.flc * 2, cfg.flc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc, cfg.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.flc, cfg.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(cfg.flc, cfg.input_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
