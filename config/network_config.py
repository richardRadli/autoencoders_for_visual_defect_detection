def network_configs(cfg):
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
                cfg.get("latent_space_dimension"),
            "img_size":
                cfg.get("crop_size") if cfg.get("crop_it") else cfg.get("img_size"),
            "input_channel":
                1
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
                cfg.get("latent_space_dimension"),
            "img_size":
                cfg.get("crop_size") if cfg.get("crop_it") else cfg.get("img_size"),
            "input_channel":
                1
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
                cfg.get("latent_space_dimension"),
            "img_size":
                cfg.get("crop_size") if cfg.get("crop_it") else cfg.get("img_size"),
            "input_channel":
                1
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
                cfg.get("latent_space_dimension"),
            "img_size":
                cfg.get("crop_size") if cfg.get("crop_it") else cfg.get("img_size"),
            "input_channel":
                1
        }
    }

    return network_config
