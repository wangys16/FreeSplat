import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    try:
        return {
            **views,
            "image": views["image"].flip(-1),
            "extrinsics": reflect_extrinsics(views["extrinsics"]),
        }
    except:
        exit(0)


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    # if torch.rand(tuple(), generator=generator) < 0.5:
    #     return example
    # try:
    #     return {
    #         **example,
    #         "context": [reflect_views(x) for x in example["context"]],
    #         "target": reflect_views(example["target"]),
    #     }
    # except:
    #     return {
    #     **example,
    #     "context": reflect_views(example["context"]),
    #     "target": reflect_views(example["target"]),
    #     }
    return example
