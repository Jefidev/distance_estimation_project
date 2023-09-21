import argparse
from collections import defaultdict
from time import time
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from trainers.base_trainer import Trainer
from dataset.motsynth import MOTSynth
from models.base_model import BaseLifter
from utils.metrics import get_metrics, print_metrics
from utils.plots import (
    average_variance_plot,
    bias_plot,
    distance_bins_plot,
    visibility_bins_plot,
)
from utils.sampler import (
    CustomSamplerTest,
    CustomSamplerTrain,
    CustomDistributedSampler,
)
from utils.utils_scripts import is_list_empty, nearness_to_z, z_to_nearness

plt.switch_backend("agg")


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class TrainerRegressor(Trainer):
    def __init__(self, model: BaseLifter, args: argparse.Namespace) -> None:
        super().__init__(model, args)

    def train(self) -> None:
        self.model.train()

        model = self.model.module if self.cnf.distributed else self.model

        losses = defaultdict(list)

        scaler = torch.cuda.amp.GradScaler(
            init_scale=(2**16) / self.cnf.accumulation_steps, enabled=self.cnf.fp16
        )
        self.optimizer.zero_grad()
        tot_bboxes = 0
        tot_acc = 0.0
        loss_fun = model.get_loss_fun()

        barrier()

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.epoch)

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}/{self.cnf.epochs}",
        ) as pbar:
            for step, sample in enumerate(self.train_loader):
                assert (
                    len(self.train_loader) / self.cnf.batch_size
                    > self.cnf.accumulation_steps
                )

                (
                    x,
                    _,
                    _,
                    video_bboxes,
                    distances,
                    _,
                    _,
                    head_coords,
                    frame_idx,
                    video_idx,
                    video_keypoints,
                    clip_clean,
                ) = sample
                if self.cnf.use_keypoints:
                    last_frame_keypoints = [k[-1] for k in video_keypoints]
                else:
                    last_frame_keypoints = None
                last_frame_bboxes = [v[-1] for v in video_bboxes]
                last_head_coords = [h[-1] for h in head_coords]

                if any(
                    map(is_list_empty, (last_frame_bboxes, video_bboxes, distances))
                ):
                    pbar.update()
                    continue

                x = x.to(self.cnf.device)

                y_true = torch.cat([d[-1] for d in distances]).to(self.cnf.device)
                y_true = z_to_nearness(y_true) if self.cnf.nearness else y_true

                with torch.autocast(
                    self.cnf.device.split(":")[0],
                    enabled=self.cnf.fp16,
                    dtype=torch.float16,
                ):
                    output = self.model(x, last_frame_bboxes, last_frame_keypoints)

                    loss = loss_fun(
                        y_pred=output, y_true=y_true, bboxes=last_frame_bboxes
                    )
                    losses["distance_loss"].append(loss.item())

                    loss = loss / self.cnf.accumulation_steps

                scaler.scale(loss).backward()
                losses["loss"].append(loss.item())

                if isinstance(output, tuple) and len(output) == 2:
                    output = output[0]
                output = output.detach().cpu().numpy().tolist()

                y_pred_z = (
                    nearness_to_z(torch.tensor(output)).numpy()
                    if self.cnf.nearness
                    else output
                )

                errors = np.abs(y_pred_z - y_true.cpu().detach().numpy())
                hits = sum(errors < 1)
                tot_bboxes += len(y_true)
                tot_acc += hits
                epoch_acc = tot_acc / tot_bboxes

                if (step + 1) % self.cnf.accumulation_steps == 0:
                    if self.cnf.grad_clip:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), self.cnf.grad_clip
                        )
                    scaler.step(self.optimizer)
                    self.optimizer.zero_grad()
                    scaler.update()
                    if self.cnf.scheduler == "cosine":
                        self.scheduler.step()

                pbar.set_postfix(
                    {
                        "alp@1": epoch_acc,
                        "loss": np.mean(losses["loss"]),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                pbar.update()
                if wandb.run:
                    wandb.log(
                        {"it_train_acc": hits / len(y_true)}
                        | {k: v[-1] for k, v in losses.items()}
                    )

                barrier()

        if wandb.run:
            wandb.log(
                {
                    "epoch": self.epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_train_acc": epoch_acc,
                    "train_it": (step + 1) * (self.epoch + 1),
                }
                | {f"epoch_train_{k}": np.mean(v) for k, v in losses.items()}
            )

    def test(self) -> None:
        self.model.eval()

        model = self.model.module if self.cnf.distributed else self.model

        t = time()
        all_errors = np.array([], dtype=np.float32)
        all_true = np.array([], dtype=np.float32)
        all_vars = np.array([], dtype=np.float32)
        all_pred = np.array([], dtype=np.float32)
        all_visibilities = np.array([], dtype=np.float32)
        all_idx = np.array([], dtype=np.int32)
        all_video_idx = np.array([], dtype=np.int32)
        all_classes = np.array([], dtype=np.int32)
        hits = 0
        tot = 0
        tot_loss = 0.0
        loss_fun = model.get_loss_fun()

        barrier()
        with tqdm(total=len(self.test_loader), desc="Testing") as pbar:
            for step, sample in enumerate(self.test_loader):
                (
                    x,
                    _,
                    _,
                    video_bboxes,
                    distances,
                    visibilities,
                    classes,
                    _,
                    frame_idx,
                    video_idx,
                    video_keypoints,
                    _,
                ) = sample

                last_frame_distances = [d[-1] for d in distances][-1]
                last_frame_visibilities = [v[-1] for v in visibilities][-1]
                last_frame_bboxes = [[v[-1] for v in video_bboxes][-1]]
                last_frame_classes = [c[-1] for c in classes][-1]
                if self.cnf.use_keypoints:
                    last_frame_keypoints = [k[-1] for k in video_keypoints]
                else:
                    last_frame_keypoints = None

                if is_list_empty(last_frame_bboxes):
                    pbar.update()
                    continue

                x = x.to(self.cnf.device)

                output = self.model(x, last_frame_bboxes, last_frame_keypoints)

                y_true = last_frame_distances.view(-1).to(self.cnf.device)
                y_true = z_to_nearness(y_true) if self.cnf.nearness else y_true
                loss = loss_fun(y_pred=output, y_true=y_true, bboxes=last_frame_bboxes)
                dists_pred, dists_vars = self.post_process(output)

                tot_loss += loss.item()
                # all_vars += dists_vars
                all_vars = np.append(all_vars, dists_vars)
                # all_visibilities += last_frame_visibilities.tolist()
                all_visibilities = np.append(
                    all_visibilities, last_frame_visibilities.tolist()
                )
                all_classes = np.append(all_classes, last_frame_classes.tolist())
                dists_pred = (
                    nearness_to_z(np.asarray(dists_pred)).tolist()
                    if self.cnf.nearness
                    else dists_pred
                )

                all_pred = np.append(all_pred, dists_pred)
                errors = np.abs(np.array(last_frame_distances) - np.array(dists_pred))
                all_errors = np.append(all_errors, errors)
                all_true = np.append(all_true, last_frame_distances)
                all_idx = np.append(
                    all_idx, [int(frame_idx[-1][0])] * len(last_frame_distances)
                )
                all_video_idx = np.append(
                    all_video_idx, [int(video_idx[0])] * len(last_frame_distances)
                )
                hits += len(errors[errors < 1])
                tot += len(errors)

                pbar.update()
                pbar.set_postfix({"alp@1": hits / tot, "loss": tot_loss / (step + 1)})

        all_results = {
            "all_errors": all_errors,
            "all_true": all_true,
            "all_vars": all_vars,
            "all_pred": all_pred,
            "all_visibilities": all_visibilities,
            "all_classes": all_classes,
            "all_idx": all_idx,
            "all_video_idx": all_video_idx,
        }
        self.save_results(
            filename=f"epoch_{self.epoch}_test_results.npz", results=all_results
        )

        acc = hits / tot
        current_error_mean = np.mean(all_errors)
        if self.cnf.scheduler == "plateau":
            self.scheduler.step(current_error_mean)

        error_std = np.std(all_errors)

        metrics = get_metrics(
            y_true=all_true,
            y_pred=all_pred,
            y_visibilities=all_visibilities,
            y_classes=all_classes,
            long_range=self.cnf.long_range,
        )
        current_rmse = metrics["all"]["rmse_linear"]

        if wandb.run:
            wandb.log(
                {
                    "acc": acc,
                    "error_mean": current_error_mean,
                    "error_std": error_std,
                    "test_loss": tot_loss / len(self.test_loader),
                    "epoch": self.epoch,
                    "test_it": (step + 1) * (self.epoch + 1),
                }
                | metrics["all"]
                | metrics
            )
            bias_plot(pred_dists=all_pred, true_dists=all_true)
            distance_bins_plot(all_errors=all_errors, all_true=all_true, acc=acc)
            visibility_bins_plot(
                all_visibilities=all_visibilities, all_errors=all_errors, acc=acc
            )
            if len(all_vars) > 0:
                average_variance_plot(all_vars, all_true)
            plt.close("all")

        # save best model
        if (
            self.best_test_rmse_linear is None
            or current_rmse < self.best_test_rmse_linear
        ):
            self.best_test_rmse_linear = current_rmse
            if self.cnf.distributed and self.cnf.rank == 0:
                self.model.module.save_w(self.log_path / "best.pth")
            else:
                self.model.save_w(self.log_path / "best.pth", self.cnf)
            self.patience = self.cnf.max_patience
            self.save_results(
                filename=f"best_rmse_test_results.npz", results=all_results
            )
        else:
            self.patience -= 1

        if wandb.run:
            wandb.log(
                {
                    "patience": self.patience,
                    "epoch": self.epoch,
                    "best_rmse": self.best_test_rmse_linear,
                }
            )

        if self.cnf.rank == 0:
            print(
                f"\r \t● (ACC) on TEST-set: "
                f"({acc:.2%}) "
                f"│ P: {self.patience} / {self.cnf.max_patience} "
                f"| Error Mean: {current_error_mean.item():.2f} ± {error_std.item():.2f} "
                f"| RMSE Lin: {current_rmse.item():.2f} "
                f"| Loss: {tot_loss / len(self.test_loader):.4f} "
                f"| T: {time() - t:.2f} s "
            )

            print_metrics(metrics)

        barrier()

    def post_process(self, output):
        if isinstance(output, tuple):
            # with variance
            dists_pred = output[0].detach().cpu().numpy()
            dists_vars = output[1].detach().cpu().numpy()
        else:
            # without variance
            dists_pred = output.detach().cpu().numpy()
            dists_vars = []
        return dists_pred, dists_vars

    def get_dataset(self, args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
        training_set = MOTSynth(args)
        test_set = MOTSynth(args, mode="test")

        if not self.cnf.test_only:
            if self.cnf.distributed:
                train_sampler = CustomDistributedSampler(
                    training_set, stride=args.train_sampling_stride
                )
            else:
                train_sampler = CustomSamplerTrain(
                    training_set, self.cnf.seed, stride=args.train_sampling_stride
                )
            self.train_loader = DataLoader(
                dataset=training_set,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                pin_memory=True,
                worker_init_fn=training_set.wif,
                drop_last=True,
                collate_fn=training_set.collate_fn,
                sampler=train_sampler,
            )

        if not self.cnf.distributed:
            test_sampler = CustomSamplerTest(test_set, stride=args.test_sampling_stride)
        else:
            test_sampler = CustomDistributedSampler(
                test_set, train=False, stride=args.test_sampling_stride
            )
        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            pin_memory=True,
            worker_init_fn=test_set.wif_test,
            num_workers=args.n_workers,
            drop_last=True,
            sampler=test_sampler,
        )
        return training_set, test_set
