import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDatasetDenoising(Dataset):
    def __init__(self, root_dir, noise_dir, grayscale):
        self.root_dir = root_dir
        self.noise_dir = noise_dir
        self.image_files = sorted([os.path.join(root_dir, filename) for filename in os.listdir(root_dir)])
        self.noise_files = sorted([os.path.join(noise_dir, filename) for filename in os.listdir(noise_dir)])

        assert len(self.image_files) == len(self.noise_files), "Number of image files and noise files must be the same"

        if grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        noise_path = self.noise_files[idx]

        try:
            # Load images
            image = Image.open(image_path)
            noise_image = Image.open(noise_path)

            # Apply transformations
            if self.transform:
                image = self.transform(image)
                noise_image = self.transform(noise_image)

            return image, noise_image

        except Exception as e:
            print(f"Error loading image at path {image_path} or {noise_path}: {e}")
            return torch.zeros(1, 224, 224), torch.zeros(1, 224, 224)
