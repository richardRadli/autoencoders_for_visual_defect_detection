import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([os.path.join(root_dir, filename) for filename in os.listdir(root_dir)])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
