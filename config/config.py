import argparse


class ConfigTraining:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--validation_split', type=float, default=0.2)
        self.parser.add_argument("--dataset_type", type=str, default="bottle", choices=["bottle", "cable"])
        self.parser.add_argument('--network_type', type=str, default='EXTENDED', choices=['BASE', 'EXTENDED'])
        self.parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        self.parser.add_argument('--loss_function_type', type=str, default='ssim', choices=['mse', 'ssim'])
        self.parser.add_argument('--step_size', type=int, default=10, help='step size')
        self.parser.add_argument('--gamma', type=float, default=1e-5)
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='')
        self.parser.add_argument('--num_crops', type=int, default=10, help='number of crops')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
