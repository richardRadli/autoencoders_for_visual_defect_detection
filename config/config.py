import argparse


class ConfigAugmentation:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--do_augmentation', type=bool, default=True)
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "texture_2"])
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='size of the crops from the image')
        self.parser.add_argument('--augment_num', type=int, default=10000, help='number of crops to make')
        self.parser.add_argument('--p_rotate', type=float, default=0.3, help='probability of rotation')
        self.parser.add_argument('--rotate_angle_vari', type=float, default=45.0, help='amount of rotation in degrees')
        self.parser.add_argument('--p_rotate_crop', type=float, default=1.0, help='probability of rotation crop')
        self.parser.add_argument('--p_crop', type=int, default=1, choices=[0, 1], help='crop or not')
        self.parser.add_argument('--p_horizontal_flip', type=float, default=0.3, help='probability of horizontal flip')
        self.parser.add_argument('--p_vertical_flip', type=float, default=0.3, help='probability of vertical flip')
        self.parser.add_argument('--size_of_cover', type=int, default=20,
                                 help='size of the rectangles covering the image')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class ConfigTraining:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--validation_split', type=float, default=0.2, help='split rate of the dataset')
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "texture_2"])
        self.parser.add_argument('--network_type', type=str, default='AE', choices=['AE', 'AEE', 'DAE', 'DAEE'])
        self.parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--loss_function_type', type=str, default='ssim', choices=['mse', 'ssim'])
        self.parser.add_argument('--decrease_learning_rate', type=bool, default=False, choices=[True, False],
                                 help='decrease learning or not')
        self.parser.add_argument('--step_size', type=int, default=10, help='epoch rate of changing the learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.5, help='amount of changing the learning rate')
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_it', type=bool, default=True, help='crop the image or not')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='size of the crops from the image')
        self.parser.add_argument('--latent_space_dimension', type=int, default=100,
                                 help='dimension of the latent space')
        self.parser.add_argument('--vis_during_training', type=bool, default=True,
                                 help='visualize the model performance during training on the training images')
        self.parser.add_argument('--vis_interval', type=int, default=10, help='rate of visualisation in training')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class ConfigTesting:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--network_type', type=str, default='AE', choices=['AE', 'AEE', 'DAE', 'DAEE'])
        self.parser.add_argument("--dataset_type", type=str, default="texture_1", choices=["texture_1", "texture_2"])
        self.parser.add_argument("--vis_results", type=bool, default=True)
        self.parser.add_argument('--img_size', type=tuple, default=(256, 256), help='image size')
        self.parser.add_argument('--crop_size', type=tuple, default=(128, 128), help='size of the crops from the image')
        self.parser.add_argument('--stride', type=int, default=32, help='size of stride')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
