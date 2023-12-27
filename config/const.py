import logging
import os

from utils.utils import setup_logger


class _Const(object):
    setup_logger()

    user = os.getlogin()
    root_mapping = {
        'ricsi': {
            "PROJECT_ROOT": 'D:/AE/storage/',
            "DATASET_ROOT": 'D:/mvtec/'
        }
    }

    if user in root_mapping:
        root_path = root_mapping[user]
        PROJECT_ROOT = root_path["PROJECT_ROOT"]
        DATASET_ROOT = root_path["DATASET_ROOT"]
    else:
        raise ValueError(f"Wrong user name: {user}!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.
        :param dirs: These are the directories that the function checks.
        :param root_type: Either PROJECT or DATASET.
        :return: None
        """

        for _, path in dirs.items():
            if root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


class Data(_Const):
    dirs_data = {
        "texture_1_model_weights_dir":
            "weights/texture_1_model_weights",
        "texture_1_log_dir":
            "logs/texture_1_model_logs",

        "texture_2_model_weights_dir":
            "weights/texture_2_model_weights",
        "texture_2_log_dir":
            "logs/texture_2_model_logs"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_data.get(key, ""))


class Datasets(_Const):
    dirs_dataset = {
        # B O T T L E
        "texture_1_train":
            "texture_1/train/good",

        "texture_1_aug":
            "texture_1/aug",

        "texture_1_test_defective":
            "texture_1/test/defective",

        "texture_1_gt_defective":
            "texture_1/ground_truth/defective",


        # C A B L E
        "texture_2_train":
            "texture_2/train/good",

        "texture_2_aug":
            "texture_2/aug",

        "texture_2_test_defective":
            "texture_2/test/defective",

        "texture_2_gt_defective":
            "texture_2/ground_truth/defective",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset.get(key, ""))


CONST: _Const = _Const()
# IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()