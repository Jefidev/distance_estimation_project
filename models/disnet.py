import torch
import torch.nn as nn
from trainers.base_trainer import Trainer
from dataset.naive_features import DisNetDataset
from models.base_model import BaseLifter
from trainers.trainer_mlp import TrainerMLP
from torch.utils.data import Dataset


class DisNet(BaseLifter):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(6, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.distance_estimator(x).squeeze(-1)

    def get_loss_fun(self) -> nn.Module:
        return nn.L1Loss()

    def get_trainer(self) -> Trainer:
        return TrainerMLP

    def get_dataset(self, args) -> Dataset:
        return DisNetDataset

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)
