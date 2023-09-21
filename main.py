import argparse
import os
import random
import socket
import uuid
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.backbones import BACKBONES
from models.bb_regressors import REGRESSORS
from models.disnet import DisNet
from models.temporal_compression import TEMPORAL_COMPRESSORS
from models.zhu import ZHU
from utils.distributed import init_distributed
from utils.metrics import RED, END, YELLOW

DEFAULT_DS_STATS = {
    "h_mean": 118.69,
    "w_mean": 50.21,
    "d_mean": 29.71,
    "h_std": 112.60,
    "w_std": 58.77,
    "d_std": 18.99,
    "d_95_percentile": 71.58,
}

MODELS = {
    "zhu": partial(ZHU, enhanced=False),
    "zhu_enhanced": partial(ZHU, enhanced=True),
    "disnet": DisNet,
}

if Path("/nas").exists():
    DEFAULT_DS_PATH = "/nas/softechict-nas-3/matteo/Datasets/MOTSynth"
else:
    DEFAULT_DS_PATH = f"{Path.home()}/softechict-nas-3/matteo/Datasets/MOTSynth"


def check_positive(strictly_positive: bool = True):
    def check_positive_inner(value):
        try:
            ivalue = int(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"{value} is not an integer") from e
        if strictly_positive:
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(
                    f"{value} is an invalid strictly positive int value"
                )
        else:
            if ivalue < 0:
                raise argparse.ArgumentTypeError(
                    f"{value} is an invalid positive int value"
                )
        return ivalue

    return check_positive_inner


def set_seed(seed: Optional[int] = None) -> int:
    """
    set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


def parse_args(default: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--wandb_tag", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from exp_log_path",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--save_results", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--use_keypoints",
        type=bool,
        default=False,
        help="Merge keypoints information to feature space",
    )
    parser.add_argument(
        "--use_gcn",
        type=bool,
        default=False,
        help="Use GCN to estimate distance",
    )

    # PATHS
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--ds_path", type=str, default=DEFAULT_DS_PATH)
    parser.add_argument(
        "--annotations_path", type=str, default="npy_annotations/annotations_clean"
    )
    parser.add_argument("--exp_log_path", type=str, default=None)

    # MODEL
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=("adam", "sgd")
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=("cosine", "plateau", "none")
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=check_positive(), default=4)
    parser.add_argument(
        "--model", type=str, choices=MODELS.keys(), default="multiframe_regressor"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="gaussian",
        choices=("l1", "mse", "laplacian", "gaussian"),
    )
    parser.add_argument("--nearness", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--random_crop", action="store_true", default=False)
    parser.add_argument("--fpn_hdim", type=int, default=256)
    parser.add_argument(
        "--fpn_idx_lstm",
        type=str,
        default="none",
        choices=("none", "c3", "c4", "c5", "p3", "p4", "p5"),
    )
    parser.add_argument("--use_geom", action="store_true", default=False)
    parser.add_argument("--roi_op", type=str, default="pool", choices=("align", "pool"))

    # Backbone
    parser.add_argument(
        "--backbone", type=str, default="resnetfpn34", choices=BACKBONES.keys()
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        default=False,
        help="Stop at Layer 3 (only Resnet)",
    )
    parser.add_argument(
        "--pretrain", default="imagenet", type=str, choices=("none", "imagenet")
    )
    parser.add_argument("--use_centers", action="store_true", default=True)
    parser.add_argument("--no_use_centers", dest="use_centers", action="store_false")
    parser.add_argument("--residual_centers", action="store_true", default=False)
    parser.add_argument(
        "--bam_centers",
        action="store_true",
        default=False,
        help="Gives the centers also in input to the BAM backbone",
    )

    # Temporal Compression
    parser.add_argument(
        "--temporal_compression",
        type=str,
        default="identity",
        choices=TEMPORAL_COMPRESSORS.keys(),
    )

    # Regressor
    parser.add_argument(
        "--regressor", type=str, default="roi_pooling", choices=REGRESSORS.keys()
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        nargs="+",
        default=(8, 8),
    )
    parser.add_argument(
        "--adjacency",
        type=str,
        default="iou",
        choices=("distance", "iou"),
        help="Adjacency matrix type (only for ROI GCN",
    )
    parser.add_argument(
        "--alpha_zhu", type=float, default=1e-2, help="Zhu enhanced loss weight"
    )

    # Non-Local Block
    parser.add_argument(
        "--phi_ksize",
        type=int,
        default=1,
        choices=(1, 3),
        help="[NLB] Kernel size for phi",
    )
    parser.add_argument(
        "--batch_norm_nlb",
        action="store_true",
        default=True,
        help="[NLB] Use batch norm in NLB",
    )
    parser.add_argument(
        "--no_batch_norm_nlb", dest="batch_norm_nlb", action="store_false"
    )
    parser.add_argument(
        "--sub_sample_nlb",
        action="store_true",
        default=False,
        help="[NLB] Use sub-sampling in NLB",
    )
    parser.add_argument(
        "--nlb_mode",
        type=str,
        default="gaussian",
        choices=("gaussian", "dot", "embedded"),
        help="[NLB] Mode for NLB",
    )

    # Inference
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Test only without training",
    )

    # DATASET
    parser.add_argument("--use_debug_dataset", action="store_true", default=False)
    parser.add_argument("--test_sampling_stride", type=check_positive(), default=400)
    parser.add_argument("--train_sampling_stride", type=check_positive(), default=50)
    parser.add_argument("--c_map_loss_mul", type=float, default=100.0)
    parser.add_argument("--h_map_loss_mul", type=float, default=1.0)
    parser.add_argument("--w_map_loss_mul", type=float, default=1.0)
    parser.add_argument("--input_h_w", type=int, nargs="+", default=(720, 1280))
    parser.add_argument("--min_bbox_hw", type=int, nargs="+", default=(8, 8))
    parser.add_argument(
        "--min_visibility", type=float, default=0.18
    )  # 0.18 \approx 4/22
    parser.add_argument("--crop_range", type=float, nargs="+", default=(0.0, 0.25))
    parser.add_argument(
        "--crop_mode", type=str, default="random", choices=("random", "center")
    )
    parser.add_argument("--clip_len", type=int, default=1)
    parser.add_argument("--max_stride", type=int, default=8)
    parser.add_argument(
        "--stride_sampling",
        type=str,
        default="fixed",
        choices=("fixed", "normal", "uniform"),
    )
    parser.add_argument(
        "--sampling", type=str, default="naive", choices=("naive", "smart")
    )
    parser.add_argument("--scale_factor", type=float, default=0.5)
    parser.add_argument("--use_heads", action="store_true", default=False)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--ds_stats", type=dict, default=DEFAULT_DS_STATS)
    parser.add_argument("--clip_dists", action="store_true", default=False)
    parser.add_argument("--augment_centers", action="store_true", default=False)
    parser.add_argument("--augment_centers_std", type=float, default=1)
    parser.add_argument("--aug_gaussian_blur_kernel", type=int, default=7)
    parser.add_argument(
        "--aug_gaussian_blur_sigma", nargs="+", type=float, default=(2.0, 4.0)
    )
    parser.add_argument("--noisy_bb", action="store_true")
    parser.add_argument("--long_range", action="store_true", default=False)
    parser.add_argument("--use_centers_gt", action="store_true", default=True)

    return parser.parse_args([] if default else None)


def print_project_info(args: argparse.Namespace) -> None:
    HOSTNAME = socket.gethostname()
    project_name = str(Path.cwd().stem)
    m_str = f"┃ {project_name}@{HOSTNAME} ┃"
    u_str = "┏" + "━" * (len(m_str) - 2) + "┓"
    b_str = "┗" + "━" * (len(m_str) - 2) + "┛"
    print(u_str + "\n" + m_str + "\n" + b_str)
    # print(f'\n{args}')
    if args.use_debug_dataset:
        print("\n", "\033[93m", "*" * 10, "Using debug dataset \033[0m")

    print(f"\n▶ Starting Experiment '{args.exp_name}' [seed: {args.seed}]")


def set_default_args(args) -> argparse.Namespace:
    if isinstance(MODELS[args.model], partial):
        temporal = MODELS[args.model].func.TEMPORAL
    else:
        temporal = MODELS[args.model].TEMPORAL

    if not temporal and args.clip_len > 1:
        print(
            YELLOW
            + "The selected model does not support temporal data, falling back to clip_len=1"
            + END
            + "\n"
        )
    args.clip_len = args.clip_len if temporal else 1

    if "fpn" in args.backbone and args.fpn_idx_lstm != "none":
        args.temporal_compression = "none"
    if args.model == "zhu":
        args.fpn_bottleneck_lstm = False
        args.fpn_idx_lstm = "none"
    return args


def set_exp_name(args: argparse.Namespace):
    if args.exp_name is None:
        random_id = uuid.uuid4().hex[:6]
        if args.model in ("disnet", "mlp", "svr"):
            args.exp_name = f"{args.model}_{random_id}"
        else:
            args.exp_name = f"{args.model}_{args.backbone}_{args.regressor}_{args.input_h_w[0]}x{args.input_h_w[1]}_{random_id}"
    else:
        if args.resume:
            assert (
                args.resume and args.exp_name is not None
            ), "Cannot resume without --exp_name"


def preprocess_args(args):
    assert Path(args.ds_path).exists(), f"Dataset path {args.ds_path} does not exist"

    if not Path(args.annotations_path).exists():
        if Path(args.ds_path, args.annotations_path).exists():
            args.annotations_path = Path(args.ds_path, args.annotations_path)
        else:
            raise FileNotFoundError(
                f"Annotations path {args.annotations_path} does not exist"
            )

    if args.debug:
        args.use_debug_dataset = True
        args.wandb = False

    if args.nearness:
        args.ds_stats["d_mean"] = -3.0915960980110233
        args.ds_stats["d_std"] = 0.7842383240216794

    args.seed = set_seed(args.seed)

    args = set_default_args(args)

    init_distributed(args)

    if args.device is None:
        args.device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"

    set_exp_name(args)
    print_project_info(args)

    print("Dataset path: ", Path(args.ds_path).resolve())
    print("Annotations path: ", Path(args.annotations_path).resolve())

    if args.exp_log_path is None:
        args.project_log_path = Path(args.log_path)
        args.exp_log_path = args.project_log_path / args.exp_name

    args.exp_log_path.mkdir(parents=True, exist_ok=True)
    return args


def main(args: argparse.Namespace) -> None:
    args = preprocess_args(args)

    assert not (
        args.resume and args.checkpoint
    ), "Cannot use both resume and checkpoint"

    model = MODELS[args.model](args)
    trainer = model.get_trainer()(model, args)

    trainer.run()


if __name__ == "__main__":
    main(args=parse_args())
