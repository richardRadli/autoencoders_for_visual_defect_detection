import cv2
import glob
import random
import os

from colorthief import ColorThief
from tqdm import tqdm


def main():
    size_of_cover = 35
    size_of_image = 128
    name = "bottle"

    path_good = "D:/research/datasets/mvtec/%s/train/train_patches" % name
    path_covered = "D:/research/datasets/mvtec/%s/train/train_patches_covered_256_128_%dx%d" % \
                   (name, size_of_cover, size_of_cover)

    if os.path.isdir(path_covered):
        print("In the making...")
        print(path_covered)
    else:
        print("no dir exits, let me create it")
        os.mkdir(path_covered)
        print(path_covered)
        print("In the making...")

    images_good = sorted(glob.glob(path_good + "/*.png"))
    for image_good in tqdm(images_good):
        img_good = cv2.imread(image_good, 1)
        name = image_good.split("\\")[1]
        name = name.split(".")[0]

        color_thief = ColorThief(image_good)
        dominant_color = color_thief.get_color(quality=1)

        rand = random.randint(0, (size_of_image - size_of_cover))

        image_covered = cv2.rectangle(img_good, (rand, rand), (rand + size_of_cover, rand + size_of_cover),
                                      dominant_color, -1)
        cv2.imwrite(path_covered + "/" + name + ".png", image_covered)


if __name__ == "__main__":
    main()
