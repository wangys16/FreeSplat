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
from torch.utils.data import IterableDataset, Dataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

import numpy as np
import os
from PIL import Image

from torch.utils.data import get_worker_info
from torch.distributed import init_process_group, get_rank, get_world_size


@dataclass
class DatasetScannetCfg(DatasetCfgCommon):
    name: Literal["scannet"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    load_depth: bool = False
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
        if self.data_stage not in ['test', 'test_fvs']:
            for root in cfg.roots:
                root = root / self.data_stage
                root_chunks = sorted(
                    [path for path in root.iterdir()]
                )
                self.chunks.extend(root_chunks)
        else:
            # print('evaluation index:', self.index)
            root = cfg.roots[0] / self.data_stage
            self.chunks = sorted(
                    [root / path for path in self.index]
                )
            # print('self.chunks:', self.chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    # def __iter__(self):
    def __getitem__(self, idx):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.

        path = self.chunks[idx]
        # print('+++++++++++++++++path:', path)
        scene = str(path).split('/')[-1]
        if self.data_stage in ['test']:
            path = str(path)[:-2]

        # print('self.chunks:', self.chunks)
        # try:
        imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
        # except:
        #     print('self.chunks[idx]:', self.chunks[idx])
        # print('+++++++++++++++++++++++++++imshape:', imshape)
        # extrinsics, intrinsics = self.convert_poses(example["cameras"])
        extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
        intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        context_index, target_indices, fvs_length = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
            test_fvs=self.data_stage == 'test_fvs',
            path=path,
            )
        test_fvs = fvs_length > 0
        intrinsics[:, :1] /= imshape[2]
        intrinsics[:, 1:2] /= imshape[1]

        depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'depth', '0.png'))).shape
        depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_depth.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        depth_intrinsics[:, :1] /= depth_imshape[2]
        depth_intrinsics[:, 1:2] /= depth_imshape[1]
        example = {'context': [], 'scene': scene}
        scale = 1

        context_images = []
        context_depths = []
        for idx in context_index:
            img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
            img = self.to_tensor(img.resize((640, 480)))
            context_images.append(img[None])
            img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
            img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
            img = self.to_tensor(img)
            context_depths.append(img[None])
        context_images = torch.cat(context_images)
        context_depths = torch.cat(context_depths)
        content = {"extrinsics": extrinsics[context_index],
                    "intrinsics": intrinsics[context_index],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_index)) / scale,
                    "far": self.get_bound("far", len(context_index)) / scale,
                    "index": context_index,
                    }
        if self.cfg.load_depth:
            content['depth'] = context_depths
            content['depth_intrinsics'] = depth_intrinsics[context_index]
        example['context'].append(content)
        target_images = []
        if not test_fvs:
            for idx in target_indices:
                img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
        else:
            length = len(target_indices)
            for idx in target_indices[:length-fvs_length]:
                img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
            sign = int(path[-1])
            for idx in target_indices[length-fvs_length:]:
                img = Image.open(os.path.join(str(path), 'color', str(idx.numpy())+'.jpg'))
                # img = Image.open(os.path.join(str(path).replace(f'_0{sign}', f'_0{1-sign}'), 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
        
        
        target_images = torch.cat(target_images)
        if not test_fvs:
            example["target"] = {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                    "test_fvs": False,
                }
        else:
            length = len(target_indices)
            x = torch.from_numpy(np.load(os.path.join(str(path), 'extrinsics.npy'))).float()
            example["target"] = {
                    "extrinsics": torch.cat([extrinsics[target_indices[:length-fvs_length]],
                                             x[target_indices[length-fvs_length:]]]),
                    "intrinsics": intrinsics[torch.zeros_like(target_indices, device=target_indices.device)],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                    "test_fvs": fvs_length,
                }
        if self.cfg.load_depth:
            target_depths = []
            for idx in target_indices:
                img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
                img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
                img = self.to_tensor(img)
                target_depths.append(img[None])
                # print('path:', os.path.join(path, 'depth', str(idx.numpy())+'.png'))
            # print('context_depths', context_depths)
            target_depths = torch.cat(target_depths)
            example['target']['depth'] = target_depths
            example['target']['depth_intrinsics'] = depth_intrinsics[context_index]
        # print('example:', example)
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        example = apply_crop_shim(example, tuple(self.cfg.image_shape))
        # print('near:', example['context']['near'])
        # print('far:', example['context']['far'])
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
        # print('-----------------torch_images.shape:', torch_images.shape)
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        # print('value:', value)
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
                # Load the root's index.
                # with (root / data_stage / "index.json").open("r") as f:
                #     index = json.load(f)
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
        # print('-----------------merge_indices:', merged_index.keys())
        # print('-----------------self.view_sampler.index:', self.view_sampler.index.keys())
        if self.data_stage == 'test':
            merged_index = {k: v for k, v in merged_index.items() if k in self.view_sampler.index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())


# class DatasetScannet(IterableDataset):
    # cfg: DatasetScannetCfg
    # stage: Stage
    # view_sampler: ViewSampler

    # to_tensor: tf.ToTensor
    # chunks: list[Path]
    # near: float = 0.1
    # far: float = 1000.0

    # def __init__(
    #     self,
    #     cfg: DatasetScannetCfg,
    #     stage: Stage,
    #     view_sampler: ViewSampler,
    # ) -> None:
    #     super().__init__()
    #     self.cfg = cfg
    #     self.stage = stage
    #     self.view_sampler = view_sampler
    #     self.to_tensor = tf.ToTensor()

    #     # Collect chunks.
    #     self.chunks = []
    #     if self.data_stage != 'test':
    #         for root in cfg.roots:
    #             root = root / self.data_stage
    #             root_chunks = sorted(
    #                 [path for path in root.iterdir()]
    #             )
    #             self.chunks.extend(root_chunks)
    #     else:
    #         root = cfg.roots[0] / self.data_stage
    #         self.chunks = sorted(
    #                 [root / path for path in self.index]
    #             )
    #         # print('self.chunks:', self.chunks)
    #     if self.cfg.overfit_to_scene is not None:
    #         chunk_path = self.index[self.cfg.overfit_to_scene]
    #         self.chunks = [chunk_path] * len(self.chunks)

    # def shuffle(self, lst: list) -> list:
    #     indices = torch.randperm(len(lst))
    #     return [lst[x] for x in indices]

    # def __iter__(self):
    #     # Chunks must be shuffled here (not inside __init__) for validation to show
    #     # random chunks.
    #     worker_info = get_worker_info()
    #     if self.stage in ("train", "val"):
    #         self.chunks = self.shuffle(self.chunks)

    #     # When testing, the data loaders alternate chunks.
    #     worker_info = torch.utils.data.get_worker_info()
    #     if self.stage == "test" and worker_info is not None:
    #         self.chunks = [
    #             chunk
    #             for chunk_index, chunk in enumerate(self.chunks)
    #             if chunk_index % worker_info.num_workers == worker_info.id
    #         ]

    #     # for chunk_path in self.chunks:
    #     #     # Load the chunk.
    #     #     chunk = torch.load(chunk_path)

    #     #     if self.cfg.overfit_to_scene is not None:
    #     #         item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
    #     #         assert len(item) == 1
    #     #         chunk = item * len(chunk)

    #     #     if self.stage in ("train", "val"):
    #     #         chunk = self.shuffle(chunk)
    #     while True:
    #         for path in self.chunks:
    #             # print('+++++++++++++++++path:', path)
    #             scene = str(path).split('/')[-1]
    #             if self.data_stage == 'test':
    #                 path = str(path)[:-2]


    #             imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
    #             # print('+++++++++++++++++++++++++++imshape:', imshape)
    #             # extrinsics, intrinsics = self.convert_poses(example["cameras"])
    #             extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
    #             intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
    #                                         [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
    #             intrinsics[:, :1] /= imshape[2]
    #             intrinsics[:, 1:2] /= imshape[1]

    #             depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'depth', '0.png'))).shape
    #             depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_depth.txt'))\
    #                                         [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
    #             # print('depth_imshape:', depth_imshape)
    #             # print('depth_intrinsics:', depth_intrinsics)
    #             depth_intrinsics[:, :1] /= depth_imshape[2]
    #             depth_intrinsics[:, 1:2] /= depth_imshape[1]
    #             # print('intrinsics:', intrinsics)
    #             # print('extrinsics:', extrinsics)
    #             # print('extrinsics.shape:', extrinsics.shape)
    #             # print('intrinsics.shape:', intrinsics.shape)
    #             # scene = example["key"]

    #             # try:
    #             context_indices, target_indices = self.view_sampler.sample(
    #                 scene,
    #                 extrinsics,
    #                 intrinsics,
    #                 )
    #             # print('context_indices:', context_indices)
    #             # print('target_indices:', target_indices)
    #             # except ValueError:
    #             #     # Skip because the example doesn't have enough frames.
    #             #     continue

    #             # Skip the example if the field of view is too wide.
    #             if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
    #                 continue

    #             # Load the images.
    #             # context_images = [
    #             #     example["images"][index.item()] for index in context_indices
    #             # ]
    #             # context_images = self.convert_images(context_images)
    #             # target_images = [
    #             #     example["images"][index.item()] for index in target_indices
    #             # ]
    #             # target_images = self.convert_images(target_images)
    #             context_images = []
    #             target_images = []
    #             for t in ['context', 'target']:
    #                 for idx in eval(f'{t}_indices'):
    #                     img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
    #                     img = self.to_tensor(img.resize((640, 480)))
    #                     eval(f'{t}_images').append(img[None])
                
    #             if self.cfg.load_depth:
    #                 context_depths = []
    #                 target_depths = []
    #                 for t in ['context', 'target']:
    #                     for idx in eval(f'{t}_indices'):
    #                         img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
    #                         img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
    #                         img = self.to_tensor(img)
    #                         eval(f'{t}_depths').append(img[None])
    #                         # print('path:', os.path.join(path, 'depth', str(idx.numpy())+'.png'))
    #                 # print('context_depths', context_depths)
    #                 context_depths = torch.cat(context_depths)
    #                 target_depths = torch.cat(target_depths)
                
                
    #             context_images = torch.cat(context_images)
    #             target_images = torch.cat(target_images)
    #             # print('context_images:', context_images)
    #             # print('target_images:', target_images)

    #             # Skip the example if the images don't have the right shape.
    #             # context_image_invalid = context_images.shape[1:] != (3, 480, 640)
    #             # target_image_invalid = target_images.shape[1:] != (3, 480, 640)
    #             # if context_image_invalid or target_image_invalid:
    #             #     print(
    #             #         f"Skipped bad example {example['key']}. Context shape was "
    #             #         f"{context_images.shape} and target shape was "
    #             #         f"{target_images.shape}."
    #             #     )
    #             #     continue

    #             # Resize the world to make the baseline 1.
    #             context_extrinsics = extrinsics[context_indices]
    #             if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
    #                 a, b = context_extrinsics[:, :3, 3]
    #                 scale = (a - b).norm()
    #                 if scale < self.cfg.baseline_epsilon:
    #                     print(
    #                         f"Skipped {scene} because of insufficient baseline "
    #                         f"{scale:.6f}"
    #                     )
    #                     continue
    #                 extrinsics[:, :3, 3] /= scale
    #             else:
    #                 scale = 1
    #             # scale = 1

    #             example = {
    #                 "context": {
    #                     "extrinsics": extrinsics[context_indices],
    #                     "intrinsics": intrinsics[context_indices],
    #                     "image": context_images,
    #                     "near": self.get_bound("near", len(context_indices)) / scale,
    #                     "far": self.get_bound("far", len(context_indices)) / scale,
    #                     "index": context_indices,
    #                 },
    #                 "target": {
    #                     "extrinsics": extrinsics[target_indices],
    #                     "intrinsics": intrinsics[target_indices],
    #                     "image": target_images,
    #                     "near": self.get_bound("near", len(target_indices)) / scale,
    #                     "far": self.get_bound("far", len(target_indices)) / scale,
    #                     "index": target_indices,
    #                 },
    #                 "scene": scene,
    #             }
    #             if self.cfg.load_depth:
    #                 example['context']['depth'] = context_depths
    #                 example['context']['depth_intrinsics'] = depth_intrinsics[context_indices]
    #                 example['target']['depth'] = target_depths
    #                 example['target']['depth_intrinsics'] = depth_intrinsics[context_indices]
    #             # print('example:', example)
    #             if self.stage == "train" and self.cfg.augment:
    #                 example = apply_augmentation_shim(example)
    #             yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    # def convert_poses(
    #     self,
    #     poses: Float[Tensor, "batch 18"],
    # ) -> tuple[
    #     Float[Tensor, "batch 4 4"],  # extrinsics
    #     Float[Tensor, "batch 3 3"],  # intrinsics
    # ]:
    #     b, _ = poses.shape

    #     # Convert the intrinsics to a 3x3 normalized K matrix.
    #     intrinsics = torch.eye(3, dtype=torch.float32)
    #     intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    #     fx, fy, cx, cy = poses[:, :4].T
    #     intrinsics[:, 0, 0] = fx
    #     intrinsics[:, 1, 1] = fy
    #     intrinsics[:, 0, 2] = cx
    #     intrinsics[:, 1, 2] = cy

    #     # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    #     w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    #     w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    #     return w2c.inverse(), intrinsics

    # def convert_images(
    #     self,
    #     images: list[UInt8[Tensor, "..."]],
    # ) -> Float[Tensor, "batch 3 height width"]:
    #     torch_images = []
    #     for image in images:
    #         image = Image.open(BytesIO(image.numpy().tobytes()))
    #         torch_images.append(self.to_tensor(image))
    #     # print('-----------------torch_images.shape:', torch_images.shape)
    #     return torch.stack(torch_images)

    # def get_bound(
    #     self,
    #     bound: Literal["near", "far"],
    #     num_views: int,
    # ) -> Float[Tensor, " view"]:
    #     value = torch.tensor(getattr(self, bound), dtype=torch.float32)
    #     return repeat(value, "-> v", v=num_views)

    # @property
    # def data_stage(self) -> Stage:
    #     if self.cfg.overfit_to_scene is not None:
    #         return "test"
    #     if self.stage == "val":
    #         return "test"
    #     return self.stage

    # @cached_property
    # def index(self) -> dict[str, Path]:
    #     data_stages = [self.data_stage]
    #     if self.cfg.overfit_to_scene is not None:
    #         data_stages = ("test", "train")
        
    #     merged_index = {}
    #     for data_stage in data_stages:
    #         for root in self.cfg.roots:
    #             # Load the root's index.
    #             # with (root / data_stage / "index.json").open("r") as f:
    #             #     index = json.load(f)
    #             with open(root / f'{data_stage}_idx.txt', 'r') as f:
    #                 index = f.read().split('\n')
    #             index = {x: Path(root / data_stage / x) for x in index}

    #             # The constituent datasets should have unique keys.
    #             assert not (set(merged_index.keys()) & set(index.keys()))

    #             # Merge the root's index into the main index.
    #             merged_index = {**merged_index, **index}
    #     # print('-----------------merge_indices:', merged_index)
    #     return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
