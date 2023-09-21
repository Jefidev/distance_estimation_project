import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from trainers.base_trainer import Trainer
from models.backbones import BACKBONES
from models.base_model import BaseLifter
from models.bb_regressors import REGRESSORS
from models.losses.gaussian_nll import GNLL
from models.losses.laplacian_nll import LNLL
from models.losses.smooth_l1 import SmoothL1
from models.losses.zhu_enhanced import EnhancedZHU
from trainers.trainer_regressor import TrainerRegressor
import torch_geometric
from models.gcn import StackedGATv2


class ZHU(BaseLifter):
    TEMPORAL = False

    def __init__(self, args: argparse.Namespace, enhanced: bool = False):
        super().__init__()
        args.use_centers = False
        self.enhanced = enhanced
        self.alpha = args.alpha_zhu
        self.loss = args.loss
        self.use_keypoints = args.use_keypoints
        self.use_gcn = args.use_gcn

        assert not (
            self.enhanced and self.loss in ("gaussian", "laplacian")
        ), "Enhanced ZHU is not compatible with gaussian or laplacian loss"

        self.output_size = 2 if self.loss in ("gaussian", "laplacian") else 1

        self.backbone = BACKBONES[args.backbone](args)
        self.regressor = REGRESSORS[args.regressor](
            input_dim=self.backbone.output_size,
            pool_size=2,
            roi_op=args.roi_op,
        )

        if self.use_keypoints:
            self.keypoints_projector = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 512),
            )

        self.distance_estimator = nn.Sequential(
            nn.Linear(
                (self.backbone.output_size * 2 * 2) + self.use_keypoints * 512, 1024
            ),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size),
        )

        if self.enhanced:
            self.keypoint_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone.output_size * 2 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Tanh(),
            )

        if self.use_gcn:
            self.self_attention = StackedGATv2(
                (self.backbone.output_size * 2 * 2) + self.use_keypoints * 512,
                2,
                8,
                (self.backbone.output_size * 2 * 2) + self.use_keypoints * 512,
                0.2,
            )

    def forward(
        self, x: torch.Tensor, bboxes: torch.Tensor, keypoints: torch.Tensor
    ) -> torch.Tensor:
        W = x.shape[-1]
        x = rearrange(x, "b c 1 h w -> b c h w")
        x = self.backbone(x)

        x = self.regressor(x, bboxes, scale=x.shape[-1] / W)
        x = x.view(x.shape[0], -1)
        if self.use_keypoints:
            k = torch.from_numpy(np.vstack(keypoints)).type_as(x).to(x.device)
            k = self.keypoints_projector(k)
            x = torch.cat([x, k], axis=-1)

        if self.use_gcn:
            lens = [len(bb) for bb in bboxes]
            adj = torch.block_diag(*[torch.ones((l, l)) for l in lens])
            adj = torch_geometric.utils.dense_to_sparse(adj)[0].to(x.device)
            x = self.self_attention(x, adj)

        z = self.distance_estimator(x).squeeze(-1)
        if self.loss in ("gaussian", "laplacian"):
            mu = F.softplus(z[..., 0])
            logvar = z[..., 1]
            z = (mu, logvar)
        else:
            z = F.softplus(z)

        if self.enhanced and self.training:
            k = self.keypoint_regressor(x)
            return z, k

        return z

    def get_loss_fun(self, **kwargs) -> nn.Module:
        if self.enhanced:
            return EnhancedZHU(alpha=self.alpha, train=self.training)
        return {"l1": SmoothL1, "gaussian": GNLL, "laplacian": LNLL}[self.loss]()

    def get_trainer(self) -> Trainer:
        return TrainerRegressor


if __name__ == "__main__":
    from main import parse_args

    args = parse_args()
    args.backbone = "vgg16"

    model = ZHU(args).to(args.device)
    x = torch.rand((args.batch_size, 3, 1, 512, 1024)).to(args.device)
    y = model.forward(x)

    print(f"$> RESNET-{model.depth}")
    print(f"───$> input shape: {tuple(x.shape)}")
    print(f"───$> output shape: {tuple(y.shape)}")
