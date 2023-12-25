from glob import glob

from config.config import ConfigAugmentation
from config.network_config import dataset_images_path_selector
from utils.utils import augment_images

aug_cfg = ConfigAugmentation().parse()

if aug_cfg.do_augmentation:
    filelist = sorted(glob(dataset_images_path_selector().get(aug_cfg.dataset_type).get("train") + "/*.png"))
    augment_images(filelist=filelist,
                   cfg=aug_cfg)

