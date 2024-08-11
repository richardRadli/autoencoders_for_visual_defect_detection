from config.data_paths import DATA_PATH, DATASET_PATH, IMAGES_PATH


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
                IMAGES_PATH.get_data_path("texture_1_reconstruction_vis"),
            "metrics":
                DATA_PATH.get_data_path("texture_1_metrics")
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
                IMAGES_PATH.get_data_path("texture_2_reconstruction_vis"),
            "metrics":
                DATA_PATH.get_data_path("texture_2_metrics")
        },
        "cpu": {
            "model_weights_dir":
                DATA_PATH.get_data_path("cpu_model_weights_dir"),
            "log_dir":
                DATA_PATH.get_data_path("cpu_log_dir"),
            "training_vis":
                IMAGES_PATH.get_data_path("cpu_training_vis"),
            "roc_plot":
                IMAGES_PATH.get_data_path("cpu_roc_plot"),
            "reconstruction_images":
                IMAGES_PATH.get_data_path("cpu_reconstruction_vis"),
            "metrics":
                DATA_PATH.get_data_path("cpu_metrics")
        }
    }

    return path_to_data


def dataset_images_path_selector():
    path_to_images = \
    {
        "texture_1":
        {
            "train":
                DATASET_PATH.get_data_path("texture_1_train"),
            "aug":
                DATASET_PATH.get_data_path("texture_1_aug"),
            "noise":
                DATASET_PATH.get_data_path("texture_1_noise"),
            "test":
            {
                "defective":
                {
                    "test_images":
                        DATASET_PATH.get_data_path("texture_1_test_defective_test_images"),
                    "ground_truth":
                        DATASET_PATH.get_data_path("texture_1_test_defective_ground_truth"),
                }
            }
        },
        "texture_2":
        {
            "train":
                DATASET_PATH.get_data_path("texture_2_train"),
            "aug":
                DATASET_PATH.get_data_path("texture_2_aug"),
            "noise":
                DATASET_PATH.get_data_path("texture_2_noise"),
            "test":
            {
                "defective":
                {
                    "test_images":
                        DATASET_PATH.get_data_path("texture_2_test_defective_test_images"),
                    "ground_truth":
                        DATASET_PATH.get_data_path("texture_2_test_defective_ground_truth"),
                }
            }
        },
        "cpu":
        {
            "train":
                DATASET_PATH.get_data_path("cpu_train"),
            "aug":
                DATASET_PATH.get_data_path("cpu_aug"),
            "noise":
                DATASET_PATH.get_data_path("cpu_noise"),
            "test":
            {
                "cpua":
                {
                    "test_images":
                        DATASET_PATH.get_data_path("cpu_test_cpua_test_images"),
                    "ground_truth":
                        DATASET_PATH.get_data_path("cpu_test_cpua_ground_truth"),
                },
                "cpuc":
                {
                    "test_images":
                        DATASET_PATH.get_data_path("cpu_test_cpuc_test_images"),
                    "ground_truth":
                        DATASET_PATH.get_data_path("cpu_test_cpuc_ground_truth"),
                },
                "cpum":
                {
                    "test_images":
                        DATASET_PATH.get_data_path("cpu_test_cpum_test_images"),
                    "ground_truth":
                        DATASET_PATH.get_data_path("cpu_test_cpum_ground_truth"),
                }
            }
        }
    }

    return path_to_images
