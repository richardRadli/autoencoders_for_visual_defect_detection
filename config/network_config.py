from config.const import DATA_PATH, DATASET_PATH
from config.config import ConfigTraining

cfg = ConfigTraining().parse()


def network_configs():
    network_config = {
        'BASE': {
            'kernel_size':
                [3, 4, 8],
            'stride':
                [1, 2],
            'padding':
                [0, 1],
            'flc':
                [32, 64, 128],
            "alpha_slope":
                0.2,
            "latent_space_dimension":
                100,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        },
        "EXTENDED": {
            'kernel_size':
                [3, 4, 8],
            'stride':
                [1, 2],
            'padding':
                [0, 1],
            'flc':
                [32, 64, 128, 256],
            "alpha_slope":
                0.2,
            "latent_space_dimension":
                100,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        }
    }

    return network_config


def dataset_data_path_selector():
    path_to_data = {
        "bottle": {
            "model_weights_dir":
                DATA_PATH.get_data_path("bottle_model_weights_dir"),
            "log_dir":
                DATA_PATH.get_data_path("bottle_log_dir")
        },
        "cable": {
            "model_weights_dir":
                DATA_PATH.get_data_path("cable_model_weights_dir"),
            "log_dir":
                DATA_PATH.get_data_path("cable_log_dir")
        }
    }

    return path_to_data


def dataset_images_path_selector():
    path_to_images = {
        "bottle": {
            "train": DATASET_PATH.get_data_path("bottle_train"),
            "aug": DATASET_PATH.get_data_path("bottle_aug")
        },
        "cable": {
            "train": DATASET_PATH.get_data_path("cable_train"),
            "aug": DATASET_PATH.get_data_path("cable_aug")
        }
    }

    return path_to_images
