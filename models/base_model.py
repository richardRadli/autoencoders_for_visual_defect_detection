import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.get("input_channel"), cfg.get("flc"), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("flc"), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("flc"), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("flc") * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 2, cfg.get("flc") * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 2, cfg.get("flc") * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 4, cfg.get("flc") * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 2, cfg.get("flc"), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("z_dim"), kernel_size=8, stride=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(cfg.get("z_dim"), cfg.get("flc"), kernel_size=8, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("flc") * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 2, cfg.get("flc") * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(cfg.get("flc") * 4, cfg.get("flc") * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc") * 2, cfg.get("flc") * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(cfg.get("flc") * 2, cfg.get("flc"), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cfg.get("flc"), cfg.get("flc"), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(cfg.get("flc"), cfg.get("input_channel"), kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = nn.functional.interpolate(decoded, size=(128, 128), mode='bilinear', align_corners=False)
        return decoded
