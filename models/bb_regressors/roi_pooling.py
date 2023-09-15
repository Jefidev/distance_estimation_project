from typing import Optional
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import roi_pool, roi_align


class ROIPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2048, pool_size: int = 4,
                 scale: float = 1 / 16, detector: bool = True, roi_op=None, **kwargs):
        super().__init__()
        self.pool_size = pool_size
        self.scale = scale
        self.roi_pooling_op = roi_pool if roi_op == 'pool' else roi_align
        self.pool_window = self.pool_size ** 2 if isinstance(self.pool_size, int) else self.pool_size[0] * \
            self.pool_size[1]

        if detector:
            self.detector = nn.Sequential(
                nn.Conv2d(input_dim, 256, kernel_size=1, stride=1, padding=0),
                nn.ELU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                nn.ELU(inplace=True),
                nn.Flatten(),
                nn.Linear(128 * self.pool_window, output_dim),
                nn.ELU(inplace=True),
            )

    def forward(self, x, bboxes: list, scale: float, **kwargs) -> torch.Tensor:
        """input shape: (B, C, T, H, W),
        bboxes shape: (B, N, 4)"""
        B = x.shape[0]

        roi_indices = torch.repeat_interleave(torch.arange(0, B),
                                              torch.tensor([len(b) for b in bboxes], requires_grad=False))
        roi_bboxes = torch.cat(bboxes)
        rois = torch.cat([roi_indices[:, None].float(), roi_bboxes], dim=1).to(x.device)
        x = self.roi_pooling_op(input=x, boxes=rois.type_as(x), output_size=self.pool_size, spatial_scale=scale)

        if hasattr(self, 'detector'):
            x = self.detector(x)
        return x
