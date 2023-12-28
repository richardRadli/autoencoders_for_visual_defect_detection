import cv2
import glob
import random
import os

from colorthief import ColorThief
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.network_config import dataset_images_path_selector


def process_image(image_path, path_covered, cfg):
    img_good = cv2.imread(image_path, 1)
    name = os.path.basename(image_path)
    name = os.path.splitext(name)[0]

    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)

    rand = random.randint(0, (cfg.img_size[0] - cfg.size_of_cover))
    image_covered = cv2.rectangle(
        img_good, (rand, rand), (rand + cfg.size_of_cover, rand + cfg.size_of_cover), dominant_color, -1
    )
    file_name = os.path.join(path_covered, name + ".png")
    cv2.imwrite(file_name, image_covered)


def main():
    cfg = ConfigAugmentation().parse()
    path_good = dataset_images_path_selector().get(cfg.dataset_type).get("aug")
    path_covered = dataset_images_path_selector().get(cfg.dataset_type).get("noise")

    images_good = sorted(glob.glob(path_good + "/*.png"))

    with ProcessPoolExecutor() as executor:
        futures = []
        for image_good in images_good:
            futures.append(executor.submit(process_image, image_good, path_covered, cfg))

        for future in tqdm(futures, desc="Processing images", total=len(futures)):
            future.result()


if __name__ == "__main__":
    main()
