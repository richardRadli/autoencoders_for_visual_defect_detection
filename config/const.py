import logging
import os

from utils.utils import setup_logger


class _Const(object):
    setup_logger()

    user = os.getlogin()
    root_mapping = {
        'ricsi': {
            "PROJECT_ROOT": 'D:/AE/',
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
        return os.path.join(self.DATASET_ROOT, self.dirs_data.get(key, ""))


class Datasets(_Const):
    dirs_dataset = {
        # B O T T L E
        "bottle_train":
            "bottle/train/good",

        "bottle_test_broken_large":
            "bottle/test/broken_large",
        "bottle_test_broken_small":
            "bottle/test/broken_small",
        "bottle_test_contamination":
            "bottle/test/contamination",
        "bottle_test_good":
            "bottle/test/good",

        "bottle_gt_broken_large":
            "bottle/ground_truth/broken_large",
        "bottle_gt_broken_small":
            "bottle/ground_truth/broken_small",
        "bottle_gt_contamination":
            "bottle/ground_truth/contamination",

        # C A B L E
        "cable_train":
            "cable/train/good",

        "cable_test_bent_wire":
            "cable/test/bent_wire",
        "cable_test_cable_swap":
            "cable/test/cable_swap",
        "cable_test_combined":
            "cable/test/combined",
        "cable_test_cut_inner_insulation":
            "cable/test/cut_inner_insulation",
        "cable_test_cut_outer_insulation":
            "cable/test/cut_outer_insulation",
        "cable_test_missing_cable":
            "cable/test/missing_cable",
        "cable_test_missing_wire":
            "cable/test/missing_wire",
        "cable_test_poke_insulation":
            "cable/test/poke_insulation",
        "cable_test_good":
            "cable/test/good",

        "cable_gt_bent_wire":
            "cable/ground_truth/bent_wire",
        "cable_gt_cable_swap":
            "cable/ground_truth/cable_swap",
        "cable_gt_combined":
            "cable/ground_truth/combined",
        "cable_gt_cut_inner_insulation":
            "cable/ground_truth/cut_inner_insulation",
        "cable_gt_cut_outer_insulation":
            "cable/ground_truth/cut_outer_insulation",
        "cable_gt_missing_cable":
            "cable/ground_truth/missing_cable",
        "cable_gt_missing_wire":
            "cable/ground_truth/missing_wire",
        "cable_gt_poke_insulation":
            "cable/ground_truth/poke_insulation",


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
# DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()