from config.const import DATA_PATH, DATASET_PATH, IMAGES_PATH
from config.config import ConfigTraining

cfg = ConfigTraining().parse()


def network_configs():
    network_config = {
        'AE': {
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
                cfg.latent_space_dimension,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        },
        'DAE': {
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
                cfg.latent_space_dimension,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        },
        "AEE": {
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
                cfg.latent_space_dimension,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        },
        "DAEE": {
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
                cfg.latent_space_dimension,
            "img_size":
                cfg.crop_size if cfg.crop_it else cfg.img_size,
            "input_channel":
                3
        }
    }

    return network_config


def dataset_data_path_selector():
    path_to_data = {
        "texture_1": {
            "model_weights_dir":
                DATA_PATH.get_data_path("texture_1_model_weights_dir"),
            "log_dir":
                DATA_PATH.get_data_path("texture_1_log_dir"),
            "training_vis":
                IMAGES_PATH.get_data_path("texture_1_training_vis"),
            "roc_plot":
                IMAGES_PATH.get_data_path("texture_1_roc_plot"),
            "reconstruction_images":
                IMAGES_PATH.get_data_path("texture_1_reconstruction_vis")

        },
        "texture_2": {
            "model_weights_dir":
                DATA_PATH.get_data_path("texture_2_model_weights_dir"),
            "log_dir":
                DATA_PATH.get_data_path("texture_2_log_dir"),
            "training_vis":
                IMAGES_PATH.get_data_path("texture_2_training_vis"),
            "roc_plot":
                IMAGES_PATH.get_data_path("texture_2_roc_plot"),
            "reconstruction_images":
                IMAGES_PATH.get_data_path("texture_2_reconstruction_vis")
        }
    }

    return path_to_data


def dataset_images_path_selector():
    path_to_images = {
        "texture_1": {
            "train": DATASET_PATH.get_data_path("texture_1_train"),
            "aug": DATASET_PATH.get_data_path("texture_1_aug"),
            "noise": DATASET_PATH.get_data_path("texture_1_noise"),
            "gt": DATASET_PATH.get_data_path("texture_1_gt_defective"),
            "test": DATASET_PATH.get_data_path("texture_1_test_defective")
        },
        "texture_2": {
            "train": DATASET_PATH.get_data_path("texture_2_train"),
            "aug": DATASET_PATH.get_data_path("texture_2_aug"),
            "noise": DATASET_PATH.get_data_path("texture_2_noise"),
            "gt": DATASET_PATH.get_data_path("texture_2_gt_defective"),
            "test": DATASET_PATH.get_data_path("texture_2_test_defective")
        }
    }

    return path_to_images
