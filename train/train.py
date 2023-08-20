import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

from data_loader import MVTecDataset
from models.network_selector import NetworkFactory


class TrainAutoEncoder:
    def __init__(self):
        # Define your dataset's root directory
        root_dir = "D:/research/datasets/mvtec/bottle/train/good"

        # Create an instance of the custom ImageDataset
        dataset = MVTecDataset(root_dir=root_dir)

        # Create a DataLoader
        batch_size = 16
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        network_cfg = {
            "input_channel": 3,
            "flc": 32,
            "z_dim": 100
        }
        self.model = NetworkFactory.create_network(network_type="BASE", network_cfg=network_cfg, device="cuda")
        summary(self.model, (3, 128, 128))

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=2e-4)

    def train(self):
        for epoch in tqdm(range(10), total=10, desc="Epochs"):
            self.model.train()  # Set the model to training mode
            total_loss = 0.0

            # Example usage in a training loop
            for batch_idx, images in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Training"):
                images = images.to("cuda")  # Move images to the GPU if available
                self.optimizer.zero_grad()  # Clear gradients

                # Forward pass
                outputs = self.model(images)
                # Calculate SSIM loss
                loss = self.criterion(outputs, images)

                # Backpropagation
                loss.backward()

                # Update weights
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 9:  # Print every 10 batches
                    print(f"Epoch [{epoch+1}/{10}], Batch [{batch_idx+1}/{len(self.dataloader)}], "
                          f"Loss: {loss.item():.4f}")

            average_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{10}], Average Loss: {average_loss:.4f}")


if __name__ == "__main__":
    ae = TrainAutoEncoder()
    ae.train()
