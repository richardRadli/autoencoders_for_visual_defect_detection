import cv2
import math
import numpy as np
import os
import random

from config.config import ConfigAugmentation
from config.network_config import dataset_images_path_selector


def generate_image_list(train_data_dir, augment_num):
    filenames = os.listdir(train_data_dir)
    num_imgs = len(filenames)
    num_ave_aug = int(math.floor(augment_num/num_imgs))
    rem = augment_num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

    img_list = [
        (os.sep.join([train_data_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]

    return img_list


def random_crop(image, new_size):
    h, w = image.shape[:2]
    y = np.random.randint(0, h - new_size[0])
    x = np.random.randint(0, w - new_size[1])
    image = image[y:y + new_size[0], x:x + new_size[1]]
    return image


def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    angle %= 360
    m_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, m_rotate, (w, h))
    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = img_rotated[y0:y0 + h_crop, x0:x0 + w_crop]
    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def augment_images(filelist, aug_out_dir, cfg):
    for filepath, n in filelist:
        img = cv2.imread(filepath)
        if img.shape[:2] != cfg.img_size:
            img = cv2.resize(img, cfg.img_size)
        filename = filepath.split(os.sep)[-1]
        dot_pos = filename.rfind('.')
        imgname = filename[:dot_pos]
        ext = filename[dot_pos:]

        print('Augmenting {} ...'.format(filename))
        for i in range(n):
            img_varied = img.copy()
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i)

            if random.random() < cfg.p_rotate:
                img_varied_ = random_rotate(
                    img_varied,
                    cfg.rotate_angle_vari,
                    cfg.p_rotate_crop)
                if img_varied_.shape[0] >= cfg.crop_size[0] and img_varied_.shape[1] >= cfg.crop_size[1]:
                    img_varied = img_varied_
                varied_imgname += 'r'

            if random.random() < cfg.p_crop:
                img_varied = random_crop(
                    img_varied,
                    cfg.crop_size)
                varied_imgname += 'c'

            if random.random() < cfg.p_horizontal_flip:
                img_varied = cv2.flip(img_varied, 1)
                varied_imgname += 'h'

            if random.random() < cfg.p_vertical_flip:
                img_varied = cv2.flip(img_varied, 0)
                varied_imgname += 'v'

            output_filepath = os.sep.join([aug_out_dir, '{}{}'.format(varied_imgname, ext)])
            cv2.imwrite(output_filepath, img_varied)


def main():
    aug_cfg = ConfigAugmentation().parse()

    if aug_cfg.do_augmentation:
        img_list = generate_image_list(
            train_data_dir=dataset_images_path_selector().get(aug_cfg.dataset_type).get("train"),
            augment_num=aug_cfg.augment_num
        )

        augment_images(filelist=img_list,
                       aug_out_dir=dataset_images_path_selector().get(aug_cfg.dataset_type).get("aug"),
                       cfg=aug_cfg)


if __name__ == "__main__":
    main()
