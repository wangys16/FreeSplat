import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews
import mmcv


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    # print('post_rescale_rgb.shape:', image_new.shape)
    return rearrange(image_new, "h w c -> c h w")

def rescale_depth(
    image,
    shape,
):
    h, w = shape
    image_new = image.detach().cpu().numpy().squeeze()
    # image_new = Image.fromarray(image_new)
    # image_new = image_new.resize((w, h), Image.LANCZOS)
    # image_new = np.array(image_new) / 255
    # try:
    # print('image_new.shape:', image_new.shape)
    # print('h,w:', h,w)
    # print('image_new:', image_new)
    # np.save('test.npy', image_new)
    image_new = mmcv.imresize(image_new.astype(np.float32), (w,h), interpolation='nearest').astype(np.float16)
    # except:
    #     print('image_new.shape:', image_new.shape)
    #     print('h,w:', h,w)
    #     exit(0)
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    # print('post_rescale_depth.shape:', image_new.shape)
    return image_new

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    use_depth: bool = False,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    

    # Center-crop the image.
    # images = images[..., :, row : row + h_out, col : col + w_out]
    # if use_depth:
    #     # print(f'h_out:{h_out}, w_out:{w_out}, row:{row}, col:{col}')
    #     images = images[..., :, :h_out, :w_out]
    #     # print('USING DEPTH!!!!!!!!!!!!!!!')
    #     # images = images[..., :, row : row + h_out, col : col + w_out]

    #     # Adjust the intrinsics to account for the cropping.
    #     # intrinsics = intrinsics.clone()
    #     # print('center cropping when using depth!!!!!!!')
    #     # intrinsics[..., 0, :] *= w_in / w_out  # fx
    #     # intrinsics[..., 1, :] *= h_in / h_out  # fy
    # else:
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    # intrinsics[..., 0, :] *= w_in / w_out  # fx
    # intrinsics[..., 1, :] *= h_in / h_out  # fy
    # print('center crop ratio:', w_in / w_out, h_in / h_out)
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    # print('middle intrinsics:', intrinsics)

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depth: bool=False,
    use_depth: bool=False,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # print(f'h_out:{h_out}, h_in:{h_in}, w_out:{w_out}, w_in:{w_in}')
    assert h_out <= h_in and w_out <= w_in

    if use_depth:
        scale_factor = max(1.015*h_out / h_in, 1.015*w_out / w_in)
    else:
        scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    # assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    if not depth:
        # print('color:')
        images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    else:
        # print('depth:')
        # print('depth.shape:', images[0].shape)
        images = torch.stack([rescale_depth(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    # print('before_crop_shape:', images.shape)

    # intrinsics[..., 0, :] *= w_in / w  # fx
    # intrinsics[..., 1, :] *= h_in / h  # fy

    # print('intrinsics:', intrinsics)
    
    return center_crop(images, intrinsics, shape, use_depth=use_depth)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    # try:
    # print('views:', views)
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape, use_depth=('depth' in views.keys()))
    # except:
    #     print('views:', views)
    #     exit(0)
    results = {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }
    if 'depth' in views.keys():
        depths, depth_intrinsics = rescale_and_crop(views["depth"], views["intrinsics"], shape, depth=True, use_depth=True)
        results['depth'] = depths
        results['depth_s-1'] = depths
        for s in range(4):
            results[f'depth_s{s}'] = rescale_and_crop(views["depth"], views["intrinsics"],\
                                                    (shape[0]//(2**(s+1)), shape[1]//(2**(s+1))), depth=True, use_depth=True)[0]
        # results['depth_intrinsics'] = depth_intrinsics
    return results


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    try:
        return {
            **example,
            "context": [apply_crop_shim_to_views(x, shape) for x in example["context"]],
            "target": apply_crop_shim_to_views(example["target"], shape),
        }
    except:
        return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
        }