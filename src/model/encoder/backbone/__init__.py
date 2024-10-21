from typing import Any

from .backbone import Backbone
from .backbone_efficientnet import BackboneEfficientNet, BackboneEfficientNetCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "efficientnet": BackboneEfficientNet,
}

BackboneCfg = BackboneEfficientNetCfg


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    return BACKBONES[cfg.name](cfg, d_in)
