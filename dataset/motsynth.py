import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms

from dataset.video_dataset import VideoFrameDataset

from utils import gt
from utils.transforms import (
    ComposeBB,
    ConvertBCHWtoCBHW,
    ConvertBHWCtoBCHW,
    ConvertImageDtypeBB,
    NoisyBB,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensorBB,
)
from utils.utils_scripts import Timer, z_to_nearness
from torchvision.transforms.functional import InterpolationMode


class MOTSynth(VideoFrameDataset):
    def __init__(
        self,
        cnf: argparse.Namespace,
        mode: str = "train",
        return_only_clean: bool = False,
        seq=None,
        no_transforms: bool = False,
    ) -> None:
        super().__init__(cnf)
        self.FILENAME_LEN = 4
        self.MODE = mode
        self.min_visibility = self.cnf.min_visibility
        self.return_only_clean = return_only_clean
        self.no_transforms = no_transforms
        if self.cnf.use_debug_dataset:
            self.ANNOTATIONS_FILE = Path(
                "dataset/motsynth/motsynth_splits/motsynth_debug.txt"
            )
            self.MAX_STRIDE = 1
        else:
            self.ANNOTATIONS_FILE = Path(
                f"dataset/motsynth/motsynth_splits/motsynth_{mode}.txt"
            )
            assert (
                self.ANNOTATIONS_FILE.exists()
            ), f"File {self.ANNOTATIONS_FILE} does not exist"
            self.MAX_STRIDE = self.cnf.max_stride

        self.transform_bb, self.transform_clip = self.get_transforms(
            self.cnf, self.MODE
        )

        with Timer(f"Loading {mode} annotations", one_line=True):
            self.sequences = tuple(self.ANNOTATIONS_FILE.read_text().splitlines())
            if seq is not None:
                self.sequences = [seq]
            self.annotations = {
                seq: np.load(Path(self.annotations_path, f"{seq}.npy"))
                for seq in self.sequences
            }

            if cnf.use_keypoints:
                self.keypoints_annotations = {
                    seq: np.load(Path(self.annotations_path, "keypoints", f"{seq}.npy"))
                    for seq in self.sequences
                }

        with Timer(f"Loading {mode} videos", one_line=True):
            self._load_videos()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, list]:
        video_idx, frame_idx = self._get_vid_and_frame_idxs(idx)
        clip, video_name, frames_name = self.videos[video_idx].get_clip(frame_idx)

        if self.return_only_clean:
            clip = torch.tensor(clip) if isinstance(clip, np.ndarray) else clip
            clip = clip.permute(0, 3, 1, 2)
            if self.no_transforms:
                return clip, frames_name, video_name
            if self.MODE == "train":
                crop = transforms.RandomResizedCrop(
                    self.input_size,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                )
            else:
                crop = transforms.Compose(
                    [
                        transforms.Resize(
                            self.input_size[0], interpolation=InterpolationMode.BICUBIC
                        ),
                        transforms.CenterCrop(self.input_size),
                    ]
                )

            T = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float32),
                    crop,
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ConvertBCHWtoCBHW(),
                ]
            )
            return T(clip), frames_name, video_name
        labels, masks = self._get_labels(video_name, frames_name)

        (
            video_bboxes,
            tracking_ids,
            video_dists,
            visibilities,
            head_coords,
        ) = self.extract_gt(labels)

        if self.cnf.use_keypoints:
            video_keypoints = self._get_keypoints_labels(video_name, frames_name, masks)

        else:
            video_keypoints = []

        good_idxs = [
            torch.ones(len(video_bboxes[i]), dtype=torch.bool)
            for i in range(len(video_bboxes))
        ]
        clip, video_bboxes, video_dists, good_idxs = self.transform_bb(
            clip, video_bboxes, video_dists, good_idxs
        )

        c_maps = torch.tensor([])
        d_map = torch.zeros(
            (
                1,
                int(self.input_size[0] * self.cnf.scale_factor),
                int(self.input_size[1] * self.cnf.scale_factor),
            )
        )
        m_map = torch.zeros(
            (
                1,
                int(self.input_size[0] * self.cnf.scale_factor),
                int(self.input_size[1] * self.cnf.scale_factor),
            )
        )

        # 0.03s
        for i, img in enumerate(clip):
            if len(video_bboxes[i]) == 0:
                c_maps = torch.cat((c_maps, torch.zeros((1, *self.input_size))), dim=0)
                continue

            tracking_ids[i] = tracking_ids[i][good_idxs[i]]
            visibilities[i] = visibilities[i][good_idxs[i]]
            head_coords[i] = head_coords[i][good_idxs[i]]

            # generate target maps: center heatmap, height map, width map
            d = z_to_nearness(video_dists[i]) if self.cnf.nearness else video_dists[i]
            c_map, _, _, d_map_np, m_map_np = gt.get_gt(
                img=img,
                bboxes=video_bboxes[i],
                dists=d,
                sigma=self.cnf.sigma,
                scale_factor=self.cnf.scale_factor,
                ds_stats=self.cnf.ds_stats,
                radius=self.cnf.radius,
            )

            if self.cnf.use_centers:
                c_maps = torch.cat(
                    (c_maps, torch.tensor(c_map, dtype=torch.float).unsqueeze(0)), dim=0
                )

            if i == len(clip) - 1:
                d_map = torch.tensor(d_map_np, dtype=torch.float).unsqueeze(0)
                m_map = torch.tensor(m_map_np, dtype=torch.float).unsqueeze(0)

        clip_clean = ConvertBCHWtoCBHW()(clip)
        clip = self.transform_clip(clip)

        if self.cnf.use_centers:
            clip = torch.cat((clip, c_maps.unsqueeze(0)), dim=0)
        classes = [torch.full_like(video_dist, 3.0) for video_dist in video_dists]
        return (
            clip,
            d_map,
            m_map,
            video_bboxes,
            video_dists,
            visibilities,
            classes,
            head_coords,
            frames_name,
            video_name,
            video_keypoints,
            clip_clean,
        )

    def extract_gt(self, labels):
        # x1, y1, x2, y2, valid_flag, track_id, distance, visibility, x, y, z
        video_bboxes = [torch.from_numpy(bboxes[:, :4]) for bboxes in labels]
        tracking_ids = [torch.from_numpy(bboxes[:, 5]) for bboxes in labels]
        video_dists = [bboxes[:, 6] for bboxes in labels]
        visibilities = [torch.from_numpy(bboxes[:, 7]) for bboxes in labels]
        head_coords = [torch.from_numpy(bboxes[:, 8:11]) for bboxes in labels]
        return video_bboxes, tracking_ids, video_dists, visibilities, head_coords

    def _get_keypoints_labels(self, video_name, frames_name, masks):
        vid_labels = self.keypoints_annotations[video_name]
        video_keypoints = [
            vid_labels[masks[i]][:, 1:] / 2203.0  # hypothenuse of the image
            for i, frame in enumerate(frames_name)
        ]
        return video_keypoints

    def _get_labels(self, video_name, frames_name):
        labels = self.annotations[video_name]
        masks = []
        bboxes = []
        for frame in frames_name:
            masks.append(
                (labels[:, 0] == float(frame))
                & (labels[:, 5] == 1.0)
                & (labels[:, 8] >= self.min_visibility)
            )
            bboxes.append(labels[masks[-1]][:, 1:])
        return bboxes, masks

    def _get_vid_and_frame_idxs(self, idx):
        video_idx = np.searchsorted(self.cumulative_idxs, idx, side="right") - 1
        starting_frame = idx - self.cumulative_idxs[video_idx]
        return video_idx, starting_frame

    def get_frames_path(self, video: str) -> str:
        return f"frames/{video}/rgb"

    @ staticmethod
    def get_transforms(cnf, mode):
        if mode == "train":
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    RandomHorizontalFlip(p=0.5),
                ]
            )

            if cnf.noisy_bb:
                T_BB.transforms.append(NoisyBB())

            if cnf.random_crop:
                min_crop, max_crop = cnf.crop_range
                T_BB.transforms.append(
                    RandomCrop(
                        cnf.input_h_w,
                        crop_mode=cnf.crop_mode,
                        min_crop=min_crop,
                        max_crop=max_crop,
                    )
                )

            T_BB.transforms.append(Resize(cnf.input_h_w))

            T_CLIP = transforms.Compose(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                            )
                        ],
                        p=0.25,
                    ),
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(
                                kernel_size=cnf.aug_gaussian_blur_kernel,
                                sigma=cnf.aug_gaussian_blur_sigma,
                            )
                        ],
                        p=0.25,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
                    # transforms.RandomPosterize(bits=4, p=0.1), # supported only for uint8 imgs
                    ConvertBCHWtoCBHW(),
                ]
            )
        else:
            T_BB = ComposeBB(
                [
                    ToTensorBB(),
                    ConvertBHWCtoBCHW(),
                    ConvertImageDtypeBB(torch.float32),
                    # ConvertBCHWtoCBHW(),
                    Resize(cnf.input_h_w),
                ]
            )
            T_CLIP = transforms.Compose(
                [
                    ConvertBCHWtoCBHW(),
                ]
            )
        return T_BB, T_CLIP

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the MOTSynth dataset.

        Args:
            batch (list): List of tuples (image, targets, dists)

        Returns:
            tuple: (images, targets, dists)
        """
        clip, d_map, m_map, *other, clip_clean = zip(*batch)
        return (
            torch.stack(clip),
            torch.stack(d_map),
            torch.stack(m_map),
            *other,
            torch.stack(clip_clean),
        )
