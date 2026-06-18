from shared.core.data_paths import CONFIG_PATHS, DATASET_PATHS, STORAGE_PATHS


def config_paths(config_type) -> dict:
    cfg_paths = {
        "augmentation_config": CONFIG_PATHS["augmentation_config"],
    }
    return cfg_paths[config_type]


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

def storage_paths(dataset_type) -> dict:
    stg_paths = {
        "texture_1": {
            "model_weights": STORAGE_PATHS["texture_1_model_weights"],
            "model_logs": STORAGE_PATHS["texture_1_model_logs"],
            "metrics": STORAGE_PATHS["texture_1_metrics"],
            "training_vis": STORAGE_PATHS["texture_1_training_vis"],
            "roc_plot": STORAGE_PATHS["texture_1_roc_plot"],
            "reconstruction_vis": STORAGE_PATHS["texture_1_reconstruction_vis"],
            "reconstruction": STORAGE_PATHS["texture_1_reconstruction"],
        },
        "texture_2": {
            "model_weights": STORAGE_PATHS["texture_2_model_weights"],
            "model_logs": STORAGE_PATHS["texture_2_model_logs"],
            "metrics": STORAGE_PATHS["texture_2_metrics"],
            "training_vis": STORAGE_PATHS["texture_2_training_vis"],
            "roc_plot": STORAGE_PATHS["texture_2_roc_plot"],
            "reconstruction_vis": STORAGE_PATHS["texture_2_reconstruction_vis"],
            "reconstruction": STORAGE_PATHS["texture_2_reconstruction"],
        },
        "cpu": {
            "model_weights": STORAGE_PATHS["cpu_model_weights"],
            "model_logs": STORAGE_PATHS["cpu_model_logs"],
            "metrics": STORAGE_PATHS["cpu_metrics"],
            "training_vis": STORAGE_PATHS["cpu_training_vis"],
            "roc_plot": STORAGE_PATHS["cpu_roc_plot"],
            "reconstruction_vis": STORAGE_PATHS["cpu_reconstruction_vis"],
            "reconstruction": STORAGE_PATHS["cpu_reconstruction"],
        },
    }
    return stg_paths[dataset_type]