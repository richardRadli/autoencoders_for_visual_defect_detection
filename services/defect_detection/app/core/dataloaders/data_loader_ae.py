import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, grayscale: bool):
        self.image_files = sorted(
            os.path.join(root_dir, filename)
            for filename in os.listdir(root_dir)
            if filename.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.num_channel = "L" if grayscale else "RGB"

        if grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_files[idx]

        try:
            image = Image.open(image_path).convert(self.num_channel)
        except (IOError, OSError) as e:
            raise ValueError(f"Error loading image at path {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image