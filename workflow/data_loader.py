import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import crop


class MVTecDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.image_files = sorted([os.path.join(root_dir, filename) for filename in os.listdir(root_dir)])
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at path {image_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image)

        return image
