import cv2
import random
import os

from colorthief import ColorThief
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from config.data_paths import JSON_FILES_PATHS
from config.dataset_config import dataset_images_path_selector
from utils.utils import load_config_json, file_reader


def process_image(image_path: str, path_covered: str, cfg) -> None:
    """
    Process an image by adding a colored cover to a random region.

    :param image_path: Path to the input image.
    :param path_covered: Path to save the processed image.
    :param cfg: Configuration object.
    :return: None
    """

    img_good = cv2.imread(image_path, 1)
    name = os.path.basename(image_path)
    name = os.path.splitext(name)[0]

    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)

    rand_x = random.randint(0, (cfg.get("crop_size")[0] - cfg.get("size_of_cover")))
    rand_y = random.randint(0, (cfg.get("crop_size")[1] - cfg.get("size_of_cover")))

    image_covered = cv2.rectangle(
        img=img_good,
        pt1=(rand_x, rand_y),
        pt2=(rand_x + cfg.get("size_of_cover"), rand_y + cfg.get("size_of_cover")),
        color=dominant_color,
        thickness=-1
    )

    file_name = os.path.join(path_covered, f'{name}.png')
    cv2.imwrite(file_name, image_covered)


def main() -> None:
    """
    Main function for processing augmented images with colored covers.

    :return: None
    """

    cfg = (
        load_config_json(json_schema_filename=JSON_FILES_PATHS.get_data_path("config_schema_augmentation"),
                         json_filename=JSON_FILES_PATHS.get_data_path("config_augmentation"))
    )

    path_good = (
        dataset_images_path_selector().get(cfg.get("dataset_type")).get("aug")
    )

    path_covered = (
        dataset_images_path_selector().get(cfg.get("dataset_type")).get("noise")
    )

    images_good = file_reader(path_good, "png")

    with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        futures = []
        for image_good in images_good:
            futures.append(executor.submit(process_image, image_good, path_covered, cfg))

        for future in tqdm(futures, desc="Processing images", total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
