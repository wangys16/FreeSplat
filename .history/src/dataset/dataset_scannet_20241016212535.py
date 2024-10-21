import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

import numpy as np
import os
from PIL import Image


@dataclass
class DatasetScannetCfg(DatasetCfgCommon):
    name: Literal["scannet", "replica"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    load_depth: bool = True
    near: float = 0.5
    far: float = 15.0

class DatasetScannet(Dataset):
    cfg: DatasetScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetScannetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        if not os.path.exists(cfg.roots[0]):
            if os.path.exists('/home/wang/ssd/scannet/scannet_surn/scannet'):
                cfg.roots[0] = Path('/home/wang/ssd/scannet/scannet_surn/scannet')
            elif os.path.exists('/ssd/yswang/scannet'):
                cfg.roots[0] = Path('/ssd/yswang/scannet')
            elif os.path.exists('/dataset/yswang/data/scannet/scannet'):
                cfg.roots[0] = Path('/dataset/yswang/data/scannet/scannet')
        
        print('-'*20 + f'data root: {cfg.roots[0]}')
        if self.data_stage != 'test':
            for root in cfg.roots:
                root = root / self.data_stage
                root_chunks = sorted(
                    [path for path in root.iterdir()]
                )
                self.chunks.extend(root_chunks)
        else:
            root = cfg.roots[0] / self.data_stage
            self.chunks = sorted(
                    [root / path for path in self.index]
                )
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __getitem__(self, idx):
        path = self.chunks[idx]
        scene = str(path).split('/')[-1]
        if self.data_stage in ['test']:
            path = str(path)[:-2]

        imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
        extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
        intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        context_index, target_indices, fvs_length = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
            path=path,
            )
        intrinsics[:, :1] /= imshape[2]
        intrinsics[:, 1:2] /= imshape[1]

        depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'depth', '0.png'))).shape
        depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_depth.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        depth_intrinsics[:, :1] /= depth_imshape[2]
        depth_intrinsics[:, 1:2] /= depth_imshape[1]
        example = {'scene': scene}

        context_images = []
        context_depths = []
        for idx in context_index:
            img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
            img = self.to_tensor(img.resize((640, 480)))
            context_images.append(img[None])
            depth_img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
            depth_img = (np.asarray(depth_img.resize((640, 480))) / 1000).astype(np.float16)
            depth_img = self.to_tensor(depth_img)
            context_depths.append(depth_img[None])
        context_images = torch.cat(context_images)
        context_depths = torch.cat(context_depths)
        content = {"extrinsics": extrinsics[context_index],
                    "intrinsics": intrinsics[context_index],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_index)),
                    "far": self.get_bound("far", len(context_index)),
                    "index": context_index,
                    }
        example['context'] = content
        target_images = []

        for idx in target_indices:
            img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
            img = self.to_tensor(img.resize((640, 480)))
            target_images.append(img[None])
        
        target_images = torch.cat(target_images)

        example["target"] = {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
                "test_fvs": fvs_length,
            }
        
        if self.cfg.load_depth:
            content['depth'] = context_depths
            content['depth_intrinsics'] = depth_intrinsics[context_index]
            target_depths = []
            for idx in target_indices:
                img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
                img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
                img = self.to_tensor(img)
                target_depths.append(img[None])
            target_depths = torch.cat(target_depths)
            example['target']['depth'] = target_depths
            example['target']['depth_intrinsics'] = depth_intrinsics[context_index]
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        example = apply_crop_shim(example, tuple(self.cfg.image_shape))
        return example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        
        merged_index = {}
        for data_stage in data_stages:
            for root in self.cfg.roots:
                with open(root / f'{data_stage}_idx.txt', 'r') as f:
                    index = f.read().split('\n')
                try:
                    index.remove('')
                except:
                    pass
                index = {x: Path(root / data_stage / x) for x in index}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        if self.data_stage == 'test':
            merged_index = {k: v for k, v in self.view_sampler.index.items() if k[:-2] in merged_index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())
