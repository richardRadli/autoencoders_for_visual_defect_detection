import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MVTecDatasetDenoising(Dataset):
    def __init__(self, root_dir, noise_dir, grayscale: bool):
        self.root_dir = root_dir
        self.noise_dir = noise_dir
        self.image_files = sorted(
            os.path.join(root_dir, filename)
            for filename in os.listdir(root_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.noise_files = sorted(
            os.path.join(noise_dir, filename)
            for filename in os.listdir(noise_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        assert len(self.image_files) == len(self.noise_files), "Number of image files and noise files must be the same"

        self.num_channel = "L" if grayscale else "RGB"

        if grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        noise_path = self.noise_files[idx]

        try:
            image = Image.open(image_path).convert(self.num_channel)
            noise_image = Image.open(noise_path).convert(self.num_channel)
        except (IOError, OSError) as e:
            raise ValueError(f"Error loading image at path {image_path} or {noise_path}: {e}")

        if self.transform:
            image = self.transform(image)
            noise_image = self.transform(noise_image)

        return image, noise_image