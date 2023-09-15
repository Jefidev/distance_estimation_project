# -*- coding: utf-8 -*-
# ---------------------

from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
from utils.utils_scripts import get_local_maxima, denormalize

from utils.bboxes import BBox


RetType = Tuple[List[BBox], List[float]]
CHWMaps = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def apply_post_processing(c_map, h_map, w_map, d_map, ds_stats, scale_factor=0.25, confidence_th=0.5, device='cuda'):
    """
    :param c_map: center heatmap;
        >> torch.Tensor with shape (H/4, W/4) or (1, H/4, W/4)
    :param h_map: bbox-height map;
        >> torch.Tensor with shape (H/4, W/4) or (1, H/4, W/4)
    :param w_map: bbox-width map;
        >> torch.Tensor with shape (H/4, W/4) or (1, H/4, W/4)
    :param ds_stats: dictionary containing statistics of
        the training set, in particular:
        >> 'h_mean' and 'h_std' -> mean and std of the bbox heights
        >> 'w_mean' and 'w_std' -> mean and std of the bbox width
    :param scale_factor: the output maps (c_map, h_map, w_map)
        are smaller than the input image
        >> scale_factor = output_map.H / input_img.H
        >> values in range ]0, 1[
    :param confidence_th: detection confidence threshold; default: 0.5
    :param device: device for post-processing operations
        >> values in {'cpu', 'cuda', 'cuda:<number>'
    :return: tuple of (bboxes, confidences)
        >> bboxes: list of detected square bboxes
        >> confidence: detection confidence for each bbox:
           confidences[i] is the confidence of bboxes[i]
    """

    c_map, h_map, w_map, d_map = c_map.squeeze(), h_map.squeeze(), w_map.squeeze(), d_map.squeeze()

    # search for local maxima coordinates on the input heatmap
    # filtering peaks with values less than `confidence_th`
    centers = get_local_maxima(
        hmap=c_map, threshold=confidence_th, device=device
    )

    # calculate bbox(es) given center(s) and radius(es)
    bboxes = []
    dists = []
    confidences = []
    for c in centers:
        # rescale center coordinates
        row_index, col_index = c
        cx = int(round(col_index.item() * (1 / scale_factor)))
        cy = int(round(row_index.item() * (1 / scale_factor)))

        # read peak value -> confidence
        confidence = float(np.clip(c_map[row_index, col_index].item(), 0, 1))
        confidences.append(confidence)

        # obtain bbox height
        bbox_h = h_map[row_index, col_index].item()
        bbox_h = denormalize(
            bbox_h, mean=ds_stats['h_mean'], std=ds_stats['h_std'],
            intround=True
        )

        # obtain bbox width
        bbox_w = w_map[row_index, col_index].item()
        bbox_w = denormalize(
            bbox_w, mean=ds_stats['w_mean'], std=ds_stats['w_std'],
            intround=True
        )

        # obtain bbox dist
        bbox_d = d_map[row_index // 4, col_index // 4].item()
        bbox_d = denormalize(
            bbox_d, mean=ds_stats['d_mean'], std=ds_stats['d_std'],
            intround=True
        )

        rx, ry = bbox_w // 2, bbox_h // 2
        bboxes.append([cx - rx, cy - ry, cx + rx, cy + ry])
        dists.append(bbox_d)

    return bboxes, dists, confidences


def apply_post_processing_only_dist(bboxes, d_map, ds_stats, scale_factor=0.25, use_heads=False):
    d_map = d_map.squeeze()

    # search for local maxima coordinates on the input heatmap
    # filtering peaks with values less than `confidence_th`
    if use_heads:
        centers = [((b[0] + b[2]) / 2, b[1]) for b in bboxes]
    else:
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes]

    # calculate bbox(es) given center(s) and radius(es)
    dists = []
    for c in centers:
        # rescale center coordinates
        row_index, col_index = c
        cx = int(round(col_index * scale_factor))
        cy = int(round(row_index * scale_factor))
        cx = min(cx, d_map.shape[0] - 1)
        cy = min(cy, d_map.shape[1] - 1)

        # obtain bbox dist
        bbox_d = d_map[cx, cy].item()
        bbox_d = denormalize(
            bbox_d, mean=ds_stats['d_mean'], std=ds_stats['d_std'],
            intround=False
        )

        dists.append(bbox_d)

    return centers, dists


def post_processing(chw_maps, ds_stats, confidence_th=0.5):
    # type: (CHWMaps, Dict[str, float], float) -> RetType
    """
    :param chw_maps: tuple of (c_map, h_map, w_map), where each
        map is a torch.Tensor with shape (H/4, W/4) or (1, H/4, W/4)
    :param ds_stats: dictionary containing statistics of
        the training set, in particular:
        >> 'h_mean' and 'h_std' -> mean and std of the bbox heights
        >> 'w_mean' and 'w_std' -> mean and std of the bbox width
    :param confidence_th: detection confidence threshold; default: 0.5
        >> detection with a confidence value < `confidence_th` are ignored
    :return: tuple of (bboxes, confidences)
        >> bboxes: list of detected square bboxes
        >> confidence: detection confidence for each bbox:
           confidences[i] is the confidence of bboxes[i]
    """

    # the output maps (c_map, h_map, w_map)
    # are smaller than the input image
    # ==> scale_factor = output_map.H / input_img.H
    scale_factor = 0.25

    c_map, h_map, w_map = chw_maps
    c_map, h_map, w_map = c_map.squeeze(), h_map.squeeze(), w_map.squeeze()

    # search for local maxima coordinates on the input heatmap
    # filtering peaks with values less than `confidence_th`
    centers = get_local_maxima(
        hmap=c_map, threshold=confidence_th,
        device=c_map.device
    )

    # calculate bbox(es) given center(s) and radius(es)
    bboxes = []
    confidences = []
    for c in centers:
        # rescale center coordinates
        row_index, col_index = c
        cx = int(round(col_index.item() * (1 / scale_factor)))
        cy = int(round(row_index.item() * (1 / scale_factor)))

        # read peak value -> confidence
        confidence = float(np.clip(c_map[row_index, col_index].item(), 0, 1))
        confidences.append(confidence)

        # obtain bbox height
        bbox_h = h_map[row_index, col_index].item()
        bbox_h = denormalize(
            bbox_h, mean=ds_stats['h_mean'], std=ds_stats['h_std'],
            intround=True
        )

        # obtain bbox width
        bbox_w = w_map[row_index, col_index].item()
        bbox_w = denormalize(
            bbox_w, mean=ds_stats['w_mean'], std=ds_stats['w_std'],
            intround=True
        )

        rx, ry = bbox_w // 2, bbox_h // 2
        bboxes.append([cx - rx, cy - ry, cx + rx, cy + ry])

    return bboxes, confidences
