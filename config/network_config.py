from config.const import DATASET_PATH


def network_configs():
    network_config = {
        'BASE': {
            'kernel_size':
                [3, 4, 8],
            'stride':
                [1, 2],
            'padding':
                1,
            'flc':
                32,
            "alpha_slope":
                0.2,
            "latent_space_dimension":
                100,
            "input_channel":
                3
        }
    }

    return network_config


def dataset_images_path_selector():
    path_to_images = {
        "bottle": {
            "train": DATASET_PATH.get_data_path("bottle_train")
        },
        "cable": {
            "train": DATASET_PATH.get_data_path("cable_train")
        }
    }

    return path_to_images
