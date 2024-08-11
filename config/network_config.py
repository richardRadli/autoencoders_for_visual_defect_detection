def network_configs(cfg):
    base_network_config = {
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
            1 if cfg.get("grayscale") else 3
    }

    extended_network_config = {
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
            1 if cfg.get("grayscale") else 3
    }

    network_config = {
        "AE": base_network_config.copy(),
        "AEE": extended_network_config.copy(),
        "DAE": base_network_config.copy(),
        "DAEE": extended_network_config.copy(),
    }

    return network_config
