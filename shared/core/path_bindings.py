from shared.core.data_paths import CONFIG_PATHS, DATASET_PATHS, TRAINING_TESTING_PATHS


def config_paths() -> dict:
    cfg_paths = {
        "augmentation_config": CONFIG_PATHS["augmentation_config"],
        "testing_config": CONFIG_PATHS["testing_config"],
        "training_config": CONFIG_PATHS["training_config"],
        "tuning_config": CONFIG_PATHS["tuning_config"],
        "network_config": CONFIG_PATHS["network_config"],
    }
    return cfg_paths


def dataset_paths(dataset_type) -> dict:
    dta_paths= {
        "texture_1": {
            "good": DATASET_PATHS["texture_1_good"],
            "aug": DATASET_PATHS["texture_1_aug"],
            "noise": DATASET_PATHS["texture_1_noise"],
            "test": {
                "defective": DATASET_PATHS["texture_1_test"],
            },
            "gt": {
                "defective": DATASET_PATHS["texture_1_ground_truth"],
            },
        },
        "texture_2": {
            "good": DATASET_PATHS["texture_2_good"],
            "aug": DATASET_PATHS["texture_2_aug"],
            "noise": DATASET_PATHS["texture_2_noise"],
            "test": {
                "defective": DATASET_PATHS["texture_2_test"],
            },
            "gt": {
                "defective": DATASET_PATHS["texture_2_ground_truth"],
            },
        },
        "cpu": {
            "good": DATASET_PATHS["cpu_good"],
            "aug": DATASET_PATHS["cpu_aug"],
            "noise": DATASET_PATHS["cpu_noise"],
            "test": {
                "added": DATASET_PATHS["cpu_added_test"],
                "contamination": DATASET_PATHS["cpu_contamination_test"],
                "missing": DATASET_PATHS["cpu_missing_test"],
            },
            "gt": {
                "added": DATASET_PATHS["cpu_added_ground_truth"],
                "contamination": DATASET_PATHS["cpu_contamination_ground_truth"],
                "missing": DATASET_PATHS["cpu_missing_ground_truth"],
            },
        },
    }
    return dta_paths[dataset_type]


def training_testing_paths(dataset_type) -> dict:
    tt_paths = {
        "texture_1": {
            "model_weights": TRAINING_TESTING_PATHS["texture_1_model_weights"],
            "model_logs": TRAINING_TESTING_PATHS["texture_1_model_logs"],
            "metrics": TRAINING_TESTING_PATHS["texture_1_metrics"],
            "training_vis": TRAINING_TESTING_PATHS["texture_1_training_vis"],
            "roc_plot": TRAINING_TESTING_PATHS["texture_1_roc_plot"],
            "reconstruction_vis": TRAINING_TESTING_PATHS["texture_1_reconstruction_vis"],
            "reconstruction": TRAINING_TESTING_PATHS["texture_1_reconstruction"],
        },
        "texture_2": {
            "model_weights": TRAINING_TESTING_PATHS["texture_2_model_weights"],
            "model_logs": TRAINING_TESTING_PATHS["texture_2_model_logs"],
            "metrics": TRAINING_TESTING_PATHS["texture_2_metrics"],
            "training_vis": TRAINING_TESTING_PATHS["texture_2_training_vis"],
            "roc_plot": TRAINING_TESTING_PATHS["texture_2_roc_plot"],
            "reconstruction_vis": TRAINING_TESTING_PATHS["texture_2_reconstruction_vis"],
            "reconstruction": TRAINING_TESTING_PATHS["texture_2_reconstruction"],
        },
        "cpu": {
            "model_weights": TRAINING_TESTING_PATHS["cpu_model_weights"],
            "model_logs": TRAINING_TESTING_PATHS["cpu_model_logs"],
            "metrics": TRAINING_TESTING_PATHS["cpu_metrics"],
            "training_vis": TRAINING_TESTING_PATHS["cpu_training_vis"],
            "roc_plot": TRAINING_TESTING_PATHS["cpu_roc_plot"],
            "reconstruction_vis": TRAINING_TESTING_PATHS["cpu_reconstruction_vis"],
            "reconstruction": TRAINING_TESTING_PATHS["cpu_reconstruction"],
        },
    }
    return tt_paths[dataset_type]