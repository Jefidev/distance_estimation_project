from functools import partial

from models.backbones.efficientnet import EfficientNet
from models.backbones.mobilenet import MobileNet
from models.backbones.resnet import ResNet
from models.backbones.resnet_bam import ResNetBAM
from models.backbones.resnet_fpn import ResNetFPN
from models.backbones.shufflenet import ShuffleNet
from models.backbones.vgg import VGG16
from models.backbones.vit import ViT
from models.backbones.dino import ViTFeat

BACKBONES = {
    "resnet18": partial(ResNet, depth=18),
    "resnet34": partial(ResNet, depth=34),
    "resnet50": partial(ResNet, depth=50),
    "resnet101": partial(ResNet, depth=101),
    "resnet152": partial(ResNet, depth=152),
    "resnetfpn18": partial(ResNetFPN, depth=18),
    "resnetfpn34": partial(ResNetFPN, depth=34),
    "resnetfpn50": partial(ResNetFPN, depth=50),
    "resnetfpn101": partial(ResNetFPN, depth=101),
    "resnetfpn152": partial(ResNetFPN, depth=152),
    "resnetbam18": partial(ResNetBAM, depth=18),
    "resnetbam34": partial(ResNetBAM, depth=34),
    "resnetbam50": partial(ResNetBAM, depth=50),
    "resnetbam101": partial(ResNetBAM, depth=101),
    "resnetbam152": partial(ResNetBAM, depth=152),
    "vgg16": VGG16,
    "efficientnet-b0": EfficientNet,
    "efficientnet-b1": EfficientNet,
    "efficientnet-b2": EfficientNet,
    "efficientnet-b3": EfficientNet,
    "efficientnet-b4": EfficientNet,
    "efficientnet-b5": EfficientNet,
    "efficientnet-b6": EfficientNet,
    "efficientnet-b7": EfficientNet,
    "efficientnet_v2_s": EfficientNet,
    "efficientnet_v2_m": EfficientNet,
    "efficientnet_v2_l": EfficientNet,
    "shufflenet_v2_x0_5": ShuffleNet,
    "shufflenet_v2_x1_0": ShuffleNet,
    "shufflenet_v2_x1_5": ShuffleNet,
    "shufflenet_v2_x2_0": ShuffleNet,
    "mobilenet_v2": MobileNet,
    "mobilenet_v3_large": MobileNet,
    "mobilenet_v3_small": MobileNet,
    "vit": ViT,
    "dino": ViTFeat,
}
