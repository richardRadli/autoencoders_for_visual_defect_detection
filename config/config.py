import argparse


class ConfigAugmentation:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--do_augmentation', type=bool, default=True)
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "texture_2"])
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='')
        self.parser.add_argument('--augment_num', type=int, default=10000, help='number of crops')
        self.parser.add_argument('--p_rotate', type=float, default=0.3, help='probability of rotation')
        self.parser.add_argument('--rotate_angle_vari', type=float, default=45.0)
        self.parser.add_argument('--p_rotate_crop', type=float, default=1.0)
        self.parser.add_argument('--p_crop', type=int, default=1, choices=[0, 1])
        self.parser.add_argument('--p_horizontal_flip', type=float, default=0.3)
        self.parser.add_argument('--p_vertical_flip', type=float, default=0.3)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class ConfigTraining:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--validation_split', type=float, default=0.2)
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "cable"])
        self.parser.add_argument('--network_type', type=str, default='BASE', choices=['BASE', 'EXTENDED'])
        self.parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--loss_function_type', type=str, default='ssim', choices=['mse', 'ssim'])
        self.parser.add_argument('--decrease_learning_rate', type=bool, default=False, choices=[True, False])
        self.parser.add_argument('--step_size', type=int, default=40, help='step size')
        self.parser.add_argument('--gamma', type=float, default=1e-5)
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_it', type=bool, default=True, help='')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class ConfigTesting:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--network_type', type=str, default='BASE', choices=['BASE', 'EXTENDED'])
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "cable"])
        self.parser.add_argument("--vis_results", type=bool, default=False)
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='')
        self.parser.add_argument('--stride', type=int, default=32, help='stride')
        self.parser.add_argument('--bg_mask', type=str, default="B", choices=["B", "W"])
        self.parser.add_argument('--ssim_threshold', type=float, default=0.5, help='ssim threshold for testing')
        self.parser.add_argument('--l1_threshold', type=float, default=0.5, help='l1 threshold for testing')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
