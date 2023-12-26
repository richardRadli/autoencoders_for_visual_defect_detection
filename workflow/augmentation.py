from config.config import ConfigAugmentation
from config.network_config import dataset_images_path_selector
from utils.utils import augment_images, generate_image_list


if __name__ == "__main__":
    aug_cfg = ConfigAugmentation().parse()

    if aug_cfg.do_augmentation:
        img_list = generate_image_list(
            train_data_dir=dataset_images_path_selector().get(aug_cfg.dataset_type).get("train"),
            augment_num=aug_cfg.augment_num
        )

        augment_images(filelist=img_list,
                       aug_out_dir=dataset_images_path_selector().get(aug_cfg.dataset_type).get("aug"),
                       cfg=aug_cfg)
