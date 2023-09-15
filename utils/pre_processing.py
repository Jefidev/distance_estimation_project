# -*- coding: utf-8 -*-
# ---------------------

from typing import List
from typing import Tuple

import numpy as np

from imgaug import augmenters as iaa
import utils.bboxes
import cv2
from utils.bboxes import BBox

__MODES = ['train', 'test', 'val', 'eval']

PAD_MODES = ['constant', 'maximum', 'median', 'minimum']


class Crop(object):

    def __init__(self, min_crop_perc, max_crop_perc, target_size):
        self.min_crop_perc = min_crop_perc
        self.max_crop_perc = max_crop_perc
        self.target_h, self.target_w = target_size

    def apply(self, img, bboxes):
        h, w, _ = img.shape

        # random crop
        # left = int(round(np.random.uniform(self.min_crop_perc, self.max_crop_perc) * w))
        # right = w - int(round(np.random.uniform(self.min_crop_perc, self.max_crop_perc) * w))
        # up = int(round(np.random.uniform(self.min_crop_perc, self.max_crop_perc) * h))
        # down = h - int(round(np.random.uniform(self.min_crop_perc, self.max_crop_perc) * h))

        # # fixed crop
        # new_w = 1920 - 1280
        # new_h = 1080 - 736
        # fctr = np.random.uniform(0, 1)
        # left = int(round(fctr * new_w))
        # right = left + 1280
        # up = int(round(fctr * new_h))
        # down = up + 736

        # # center crop
        # fctr = np.random.uniform(0.0, 0.6)
        # left = int(round(fctr * 1920 // 2))
        # right = 1920 - left
        # up = int(round(fctr * 1080 // 2))
        # down = 1080 - up

        # center fixed crop
        left = (w - self.target_w) // 2
        right = w - left - 1
        up = (h - self.target_h) // 2
        down = h - up - 1
        # if left < 0:
        #     left = 0
        #     right = w - 1
        # if up < 0:
        #     up = 0
        #     down = h - 1

        if left >= 0 and right >= 0 and up >= 0 and down >= 0:

            new_bboxes = []
            for b in bboxes:
                x_min, y_min, x_max, y_max = b
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                if left < cx < right and up < cy < down:
                    new_bboxes.append([
                        x_min - left, y_min - up,
                        x_max - left, y_max - up
                    ])
                else:
                    new_bboxes.append([0, 0, 0, 0])

            return img[up:down, left:right, ...], new_bboxes

        return img, bboxes


class Resize(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply(self, img, bboxes, dists, deterministic_value=None):
        h, w, _ = img.shape
        # f = min(self.height / h, self.width / w)
        # f = min(h / self.height, w / self.width)

        if deterministic_value:
            f = deterministic_value
        else:
            f = np.random.uniform(0.6, 1.6)

        img = cv2.resize(img, (0, 0), fx=f, fy=f)

        bboxes = utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes)
        bboxes = (np.array(bboxes) * f).round().tolist()
        bboxes = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes, img)

        if dists is not None:
            dists = [d / f for d in dists]

        return img, bboxes, dists


class Augmenter:

    def __init__(self, input_size: tuple[int, int]):
        self.h, self.w = input_size

        self.crop = Crop(min_crop_perc=0, max_crop_perc=0.25, target_size=(self.h, self.w))
        self.resize = Resize(height=self.h, width=self.w)

        self.seq_pad_test = [
            iaa.PadToAspectRatio(
                aspect_ratio=self.w / self.h,
                pad_mode='constant', pad_cval=0, position='right-bottom'
            )
        ]
        self.seq_pad_train = [
            iaa.PadToFixedSize(width=self.w, height=self.h, pad_mode=PAD_MODES)
        ]

        self.seq_resize = [
            iaa.Resize(size={'height': self.h, 'width': self.w})
        ]

        # sequence of transformations to be applied to all train images
        self.seq_img = [
            iaa.Fliplr(p=0.5),
            iaa.Sometimes(0.25, iaa.GaussianBlur()),
            iaa.Sometimes(0.25, iaa.MotionBlur()),
            iaa.Sometimes(0.1, iaa.JpegCompression()),
            iaa.Sometimes(0.1, iaa.SaltAndPepper()),
            iaa.Sometimes(0.5, iaa.AddToBrightness()),
            iaa.Sometimes(0.5, iaa.AddToSaturation()),
        ]

    def apply(self, img, bboxes, dists, mode):
        # type: (np.ndarray, List[BBox], List[float], str) -> Tuple[np.ndarray, List[BBox], List[float]]
        if mode in ('test', 'val'):
            seq = iaa.Sequential(self.seq_pad_test + self.seq_resize)
            imgaug_bboxes = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes, img=img)
            img, bboxes = seq(image=img, bounding_boxes=imgaug_bboxes)
        else:
            # img, bboxes = self.crop.apply(img=img, bboxes=bboxes)
            seq_0 = iaa.Sequential(self.seq_resize)
            seq_1 = iaa.Sequential(self.seq_pad_train)
            seq_2 = iaa.Sequential(self.seq_img)

            bboxes = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes, img=img)
            img, bboxes = seq_0(image=img, bounding_boxes=bboxes)  # da Full-HD a HD
            img, bboxes, dists = self.resize.apply(img, bboxes, dists)

            bboxes = utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes_on_img=bboxes)
            img, bboxes = self.crop.apply(img=img, bboxes=bboxes)
            bboxes = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes, img=img)

            img, bboxes = seq_1(image=img, bounding_boxes=bboxes)
            img, bboxes = seq_2(image=img, bounding_boxes=bboxes)

        return img, utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes_on_img=bboxes), dists

    def apply_video(self, clip, bboxes, dists, mode):
        # type: (np.ndarray, List[BBox], List[float], str) -> Tuple[np.ndarray, List[BBox], List[float]]
        imgs = []
        bbs = []
        d = []
        if mode in ('test', 'val'):
            seq = iaa.Sequential(self.seq_pad_test + self.seq_resize).to_deterministic()
            for i, img in enumerate(clip):
                imgaug_bboxes = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes[i], img=img)
                img, bboxes[i] = seq(image=img, bounding_boxes=imgaug_bboxes)
                imgs.append(img)
                bbs.append(utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes_on_img=bboxes[i]))
            d = dists
        elif mode == 'train':
            seq_0 = iaa.Sequential(self.seq_resize).to_deterministic()
            seq_1 = iaa.Sequential(self.seq_pad_train).to_deterministic()
            seq_2 = iaa.Sequential(self.seq_img).to_deterministic()
            resize_value = np.random.uniform(0.6, 1.6)
            for i, img in enumerate(clip):
                bboxes[i] = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes[i], img=img)
                img, bboxes[i] = seq_0(image=img, bounding_boxes=bboxes[i])
                img, bboxes[i], dists[i] = self.resize.apply(img, bboxes[i], dists[i], deterministic_value=resize_value)

                bboxes[i] = utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes_on_img=bboxes[i])
                img, bboxes[i] = self.crop.apply(img=img, bboxes=bboxes[i])
                bboxes[i] = utils.bboxes.bboxes_list_to_imgaug_bboxes(bboxes_list=bboxes[i], img=img)

                img, bboxes[i] = seq_1(image=img, bounding_boxes=bboxes[i])
                img, bboxes[i] = seq_2(image=img, bounding_boxes=bboxes[i])
                imgs.append(img)
                bbs.append(utils.bboxes.imgaug_bboxes_to_bboxes_list(bboxes_on_img=bboxes[i]))
                d.append(dists[i])
        else:
            raise ValueError(f'Unknown mode: {mode}')

        return imgs, bbs, d


class PreProcessor(object):

    def __init__(self):

        self.normal = iaa.Sequential([
            iaa.PadToAspectRatio(
                aspect_ratio=self.w / self.h,
                pad_mode='constant', pad_cval=0, position='right-bottom'
            ),
            iaa.Resize(size={'height': self.h, 'width': self.w})
        ])

        self.small = iaa.Sequential([
            iaa.PadToFixedSize(
                width=self.w, height=self.h,
                pad_mode='constant', pad_cval=0, position='right-bottom'
            )
        ])

    def apply(self, img):
        h, w, _ = img.shape
        if h >= self.h or w >= self.w:
            new_img = self.normal.augment_image(img)
            new_h, new_w, _ = new_img.shape
            scale = min(new_h / h, new_w / w)
            return new_img, scale
        else:
            new_img = self.small.augment_image(img)
            return new_img, 1
