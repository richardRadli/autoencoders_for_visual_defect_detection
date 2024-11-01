from config.data_paths import JSON_FILES_PATHS


def json_config_selector(operation):
    json_cfg = {
        "augmentation": {
            "config": JSON_FILES_PATHS.get_data_path("config_augmentation"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_augmentation")
        },
        "testing": {
            "config": JSON_FILES_PATHS.get_data_path("config_testing"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_testing")
        },
        "training": {
            "config": JSON_FILES_PATHS.get_data_path("config_training"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_training")
        }
    }

    return json_cfg[operation]
