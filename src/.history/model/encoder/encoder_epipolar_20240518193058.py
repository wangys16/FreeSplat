from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone, BackboneMultiview
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .common.gaussian_adapter import Gaussians as G
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...global_cfg import get_cfg

from modules.networks import CVEncoder, DepthDecoderPP, ResnetMatchingEncoder
from modules.cost_volume import FeatureVolumeManager, CostVolumeManager, AttentionVolumeManager
from sr_utils.generic_utils import (reverse_imagenet_normalize, tensor_B_to_bM,
                                 tensor_bM_to_B)
import timm
from modules.layers import TensorFormatter
import mmcv
import torchvision.transforms as tf
from matplotlib import pyplot as plt

from .attention.transformer import LocalFeatureTransformer, GRU2D_naive_Wweights

from einops import *


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


UseDepthMode = Literal[
    "depth"
]

def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts


@dataclass
class EncoderEpipolarCfg:
    name: Literal["epipolar", "cost_volume"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    epipolar_transformer: EpipolarTransformerCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    use_epipolar_transformer: bool
    use_transmittance: bool

    est_depth: Literal["gt", "refine", "est", "cost"]


    
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    
    num_depth_candidates: int = 64
    unimatch_weights_path: str | None = "checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
    use_pc_encoder: bool = False
    
    load_depth: bool = False
    num_views: int = 2
    image_H: int = 384
    image_W: int = 512
    n_levels: int = -1
    fusion: bool = False
    op1: bool = False
    cv_type: str = 'feat'
    use_planes: bool = True
    log_plane: bool = False


    


class EncoderEpipolar(Encoder[EncoderEpipolarCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderEpipolarCfg, depth_range=[0.5, 15.0]) -> None:
        super().__init__(cfg)
        activation_func = nn.ReLU()

        self.depth_range = depth_range

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        if cfg.backbone.name == 'dino':
            self.backbone = get_backbone(cfg.backbone, 3+cfg.load_depth)
            self.backbone_projection = nn.Sequential(
                activation_func,
                nn.Linear(self.backbone.d_out, cfg.d_feature),
            )
            if cfg.use_epipolar_transformer:
                self.epipolar_transformer = EpipolarTransformer(
                    cfg.epipolar_transformer,
                    cfg.d_feature,
                )
            else:
                self.epipolar_transformer = None
            if cfg.est_depth == 'est':
                self.depth_predictor = DepthPredictorMonocular(
                    cfg.d_feature,
                    cfg.num_monocular_samples,
                    cfg.num_surfaces,
                    cfg.use_transmittance,
                )
            else:
                self.opacity_mlp = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
                self.depth_refine = nn.Sequential(
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        cfg.d_feature,
                    ),
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        1,
                    ),
                )
            if cfg.predict_opacity:
                self.to_opacity = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
            self.to_gaussians = nn.Sequential(
                activation_func,
                nn.Linear(
                    cfg.d_feature,
                    cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
                ),
            )
            
            # print('self.cfg.load_depth:', self.cfg.load_depth)
            self.high_resolution_skip = nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, cfg.d_feature, 7, 1, 3),
                activation_func,
            )
        elif cfg.backbone.name == 'cost_volume':
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )
            ckpt_path = cfg.unimatch_weights_path
            if get_cfg().mode == 'train':
                if cfg.unimatch_weights_path is None:
                    print("==> Init multi-view transformer backbone from scratch")
                else:
                    print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                    unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone.state_dict()
                        }
                    )
                    # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                    is_strict_loading = not cfg.wo_backbone_cross_attn
                    self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

            # gaussians convertor
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

            # cost volume based depth predictor
            self.depth_predictor = DepthPredictorMultiView(
                feature_channels=cfg.d_feature,
                upscale_factor=cfg.downscale_factor,
                num_depth_candidates=cfg.num_depth_candidates,
                costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
                costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                gaussians_per_pixel=cfg.gaussians_per_pixel,
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                depth_unet_attn_res=cfg.depth_unet_attn_res,
                depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                wo_depth_refine=cfg.wo_depth_refine,
                wo_cost_volume=cfg.wo_cost_volume,
                wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            )
        else:
            self.backbone = timm.create_model(
                                            "tf_efficientnetv2_s_in21ft1k", 
                                            pretrained=True, 
                                            features_only=True,
                                        )

            self.backbone.num_ch_enc = self.backbone.feature_info.channels()
            
            if cfg.use_epipolar_transformer:
                self.epipolar_transformer = EpipolarTransformer(
                    cfg.epipolar_transformer,
                    self.backbone.feature_info.channels()[0],
                )
            else:
                self.epipolar_transformer = None
            
            self.high_resolution_skip = nn.ModuleList(
                                            [nn.Sequential(
                                                nn.Conv2d(3+self.cfg.load_depth, 64, 7, 1, 3),
                                                activation_func,
                                            ),
                                            nn.Sequential(
                                                nn.Conv2d(3+self.cfg.load_depth, 64, 6, 2, 2),
                                                activation_func,
                                            ),
                                            nn.Sequential(
                                                nn.Conv2d(3+self.cfg.load_depth, 64, 8, 4, 2),
                                                activation_func,
                                            ),
                                            nn.Sequential(
                                                nn.Conv2d(3+self.cfg.load_depth, 64, 16, 8, 4),
                                                activation_func,
                                            ),
                                            nn.Sequential(
                                                nn.Conv2d(3+self.cfg.load_depth, 64, 32, 16, 8),
                                                activation_func,
                                            )]
                                        )

            self.to_gaussians = nn.Sequential(
                activation_func,
                nn.Linear(
                    64,
                    cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
                ),
            )
        

        
        
        self.gausisans_ch = cfg.num_surfaces * (2 + self.gaussian_adapter.d_in)
        # if cfg.est_depth == 'refine':
        
        self.load_depth = cfg.load_depth
        self.est_depth = cfg.est_depth
        
        if self.cfg.est_depth == 'cost':
            if not self.cfg.wo_cost_volume:
                self.matching_net = ResnetMatchingEncoder(18, 16)
                if self.cfg.cv_type == 'feat':
                    self.cost_volume = FeatureVolumeManager(matching_height=self.cfg.image_H//4, 
                                                        matching_width=self.cfg.image_W//4,
                                                        num_depth_bins=self.cfg.num_depth_candidates,
                                                        matching_dim_size=16,
                                                        num_source_views=self.cfg.num_views-1,
                                                        log_plane=self.cfg.log_plane,)
                elif self.cfg.cv_type == 'cv':
                    self.cost_volume = CostVolumeManager(matching_height=self.cfg.image_H//4, 
                                                            matching_width=self.cfg.image_W//4,
                                                            num_depth_bins=self.cfg.num_depth_candidates,
                                                            matching_dim_size=16,
                                                            num_source_views=self.cfg.num_views-1,
                                                            log_plane=self.cfg.log_plane,)
                elif self.cfg.cv_type == 'att':
                    self.cost_volume = AttentionVolumeManager(matching_height=self.cfg.image_H//4, 
                                                            matching_width=self.cfg.image_W//4,
                                                            num_depth_bins=self.cfg.num_depth_candidates,
                                                            matching_dim_size=16,
                                                            num_source_views=self.cfg.num_views-1)
                else:
                    raise ValueError(f'cv_type {self.cfg.cv_type} not recognized')
                self.cv_encoder = CVEncoder(num_ch_cv=self.cfg.num_depth_candidates,
                                            num_ch_enc=self.backbone.num_ch_enc[1:],
                                            num_ch_outs=[64, 128, 256, 384])
                dec_num_input_ch = (self.backbone.num_ch_enc[:1] 
                                                + self.cv_encoder.num_ch_enc)
            else:
                dec_num_input_ch = (self.backbone.num_ch_enc)

            self.depth_decoder = DepthDecoderPP(dec_num_input_ch, 
                                                num_output_channels=1+64,
                                                n_levels=self.cfg.n_levels,
                                                use_planes=self.cfg.use_planes,
                                                near=depth_range[0],
                                                far=depth_range[1],
                                                num_samples=self.cfg.num_depth_candidates,
                                                log_plane=self.cfg.log_plane,)

            self.tensor_formatter = TensorFormatter()

            if self.cfg.fusion:
                # self.transformer = LocalFeatureTransformer(d_model=64, 
                #                     nhead=4, layer_names=['self'], attention='linear')
                # self.blending = nn.Sequential(nn.Linear(64, 32), 
                #                             activation_func,
                #                             nn.Linear(32, 2),)
                self.weight_embedding = nn.Sequential(nn.Linear(2, 12), 
                                            activation_func,
                                            nn.Linear(12, 12),)
                self.gru = GRU2D_naive_Wweights()

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def compute_matching_feats(
                            self, 
                            cur_image, 
                            src_image, 
                            unbatched_matching_encoder_forward=False,
                        ):
        """ 
            Computes matching features for the current image (reference) and 
            source images.

            Unfortunately on this PyTorch branch we've noticed that the output 
            of our ResNet matching encoder is not numerically consistent when 
            batching. While this doesn't affect training (the changes are too 
            small), it does change and will affect test scores. To combat this 
            we disable batching through this module when testing and instead 
            loop through images to compute their feautures. This is stable and 
            produces exact repeatable results.

            Args:
                cur_image: image tensor of shape B3HW for the reference image.
                src_image: images tensor of shape BM3HW for the source images.
                unbatched_matching_encoder_forward: disable batching and loops 
                    through iamges to compute feaures.
            Returns:
                matching_cur_feats: tensor of matching features of size bchw for
                    the reference current image.
                matching_src_feats: tensor of matching features of size BMcHW 
                    for the source images.
        """
        
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_net(f) 
                                    for f in all_frames_B3hw.split(1, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                                        matching_feats, 
                                        batch_size=batch_size, 
                                        num_views=num_views,
                                    )

        else:
            # Compute matching features and batch them to reduce variance from 
            # batchnorm when training.
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1),
                apply_func=self.matching_net,
            )

        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()

        return matching_cur_feats, matching_src_feats


    def forward(
        self,
        contexts,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        is_testing: bool = False,
        test_fvs: bool = False,
        export_ply: bool = False,
    ) -> dict:
        # contexts = [contexts]
        device = contexts[0]["image"].device
        b, n_views, _, h, w = contexts[0]["image"].shape
        results = {}
        num_context_views = self.cfg.num_views


        # Encode the context images.
        if self.cfg.backbone.name == 'dino':
            context = contexts[0]
            features = self.backbone(context)
            features = rearrange(features, "b v c h w -> b v h w c")
            features = self.backbone_projection(features)
            features = rearrange(features, "b v h w c -> b v c h w")


            # Run the epipolar transformer.
            # if self.cfg.use_epipolar_transformer and (self.load_depth is None):
            if self.cfg.use_epipolar_transformer:
                # print('pre_features.shape:', features.shape)
                features, sampling = self.epipolar_transformer(
                    features,
                    context["extrinsics"],
                    context["intrinsics"],
                    context["near"],
                    context["far"],
                )
                # print('post_features.shape:', features.shape)

            # Add the high-resolution skip connection.
            if self.cfg.load_depth:
                to_skip = torch.cat([context['image'], context['depth']], dim=2)
            else:
                to_skip = context['image']
            skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip(skip)
            features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=n_views)

            h1, w1 = features.shape[-2:]

            

            # Sample depths from the resulting features.
            features = rearrange(features, "b v c h w -> b v (h w) c")
            if self.cfg.est_depth == 'est':
                depths, densities = self.depth_predictor.forward(
                    features,
                    context["near"],
                    context["far"],
                    deterministic,
                    1 if deterministic else self.cfg.gaussians_per_pixel,
                )
            else:
                depths = rearrange(context['depth'], "b v c h w -> b v (h w) c 1")
                densities = self.opacity_mlp(features).unsqueeze(-1)
                self.cfg.gaussians_per_pixel = 1



            if self.cfg.est_depth == 'refine':
                depths_raw = depths.clone()
                depths = self.depth_refine(features)[...,None]
                results['depths'] = depths
                results['depths_raw'] = depths_raw
                mask = (depths_raw > 0) * (depths_raw < 15)
                results['depths_mask'] = mask
            
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = self.to_gaussians(features)
            # print('gaussians.shape:', gaussians.shape) # [1, 2, 76800, 1, 84]
            # exit(0)
            gaussians = rearrange(
                gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            # print('gaussians.shape:', gaussians.shape) # [1, 2, 76800, 1, 84]
            # print('depths.shape:', depths.shape) # [1, 2, 76800, 1, 1]
            # exit(0)
            offset_xy = gaussians[..., :2].sigmoid()
            # print('context_extrinsics.shape:', context["extrinsics"].shape)
            # print('xy_ray.shape:', xy_ray.shape) # [196608, 1, 2]
            # print('offset_xy.shape:', offset_xy.shape) # [1, 2, 49152, 1, 2]
            # exit(0)

            if self.cfg.est_depth == 'est':
                pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
                # print('pre_xy.shape:', xy_ray.shape)
                xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                # print('post_xy.shape:', xy_ray.shape)
            else:
                xy_ray = xy_ray + torch.zeros_like(offset_xy, device=offset_xy.device)
            

            gpp = self.cfg.gaussians_per_pixel
            # print('(h,w):', (h,w))
            # exit(0)
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                (h, w),
                load_depth=self.cfg.load_depth,
            )
            # print('gaussians.means.shape:', gaussians.means.shape)
            num_gaussians = gaussians.means.shape[2] * gaussians.means.shape[1] * gaussians.means.shape[-1]
        
        elif self.cfg.backbone.name == 'cost_volume':
            # Encode the context images.
            context = contexts[0]
            if self.cfg.use_epipolar_trans:
                epipolar_kwargs = {
                    "epipolar_sampler": self.epipolar_sampler,
                    "depth_encoding": self.depth_encoding,
                    "extrinsics": context["extrinsics"],
                    "intrinsics": context["intrinsics"],
                    "near": context["near"],
                    "far": context["far"],
                }
            else:
                epipolar_kwargs = None
            trans_features, cnn_features = self.backbone(
                context["image"],
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=epipolar_kwargs,
            )

            # Sample depths from the resulting features.
            in_feats = trans_features
            extra_info = {}
            extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
            gpp = self.cfg.gaussians_per_pixel
            depths, densities, raw_gaussians = self.depth_predictor(
                in_feats,
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features,
            )

            # Convert the features and depths into Gaussians.
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.cfg.gaussians_per_pixel
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
            )

            num_gaussians = gaussians.means.shape[2] * gaussians.means.shape[1]
            
            
        elif self.cfg.est_depth == 'cost':
            # print('features.shape:', features.shape)
            # print('context_extrinsics.shape:', context["extrinsics"].shape)
            # print('context_intrinsics.shape:', context["intrinsics"].shape)
            # print('context_images.shape:', context['image'].shape)
            # exit(0)
            # depths = self.cost_depth_estimator.forward(features)

            # b, n_views, n_rays, c = features.shape
            length = len(contexts)
            gaussians = []
            coords = []
            results = {}

            self.backbone.train()

            for num in range(length):
                # with torch.no_grad() if (num < length-1) else torch.enable_grad():
                # with torch.enable_grad():
                context = contexts[num]
                context['image_shape'] = (h, w)
                self.cfg.gaussians_per_pixel = 1
                context_intrinsics = context['intrinsics'].clone()
                context_intrinsics[:,:,0] *= (w // 4)
                context_intrinsics[:,:,1] *= (h // 4)

                cur_indices = torch.arange(n_views, device=context['image'].device)
                    
                cur_intrinsics = context_intrinsics.gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,3,3))
                cur_extrinsics = context['extrinsics'].gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,4,4))
                cur_image = context['image'].gather(dim=1, index=cur_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)

                cur_feats = self.backbone(cur_image)

                resized = 0

                if not self.cfg.wo_cost_volume:

                    # print('cur_feats[0].shape:', cur_feats[0].shape)

                    if self.cfg.use_epipolar_transformer:
                        # print('pre_features.shape:', features.shape)
                        _, sampling = self.epipolar_transformer(
                            rearrange(cur_feats[0], "(b v) c h w -> b v c h w", b=b, v=n_views),
                            context["extrinsics"],
                            context["intrinsics"],
                            context["near"],
                            context["far"],
                        )
                    
                    full_indices = torch.arange(n_views, device=context['image'].device)[None].repeat(n_views,1)
                    
                    if (not test_fvs) and n_views == self.cfg.num_views:
                        src_indices = full_indices[~(full_indices == cur_indices[:,None])].view(1,n_views,n_views-1).repeat(b,1,1)
                        judge = 0
                    else:
                        judge = 1
                        slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
                        for i in range(n_views):
                            if i < num_context_views // 2:
                                start = 0
                                end = min(num_context_views, n_views)
                            elif i >= n_views - num_context_views // 2:
                                start = max(n_views - num_context_views, 0)
                                end = n_views
                            else:
                                start = max(i - (num_context_views-1) // 2, 0)
                                end = min(i + num_context_views // 2 + 1, n_views)
                            try:
                                assert end - start == min(num_context_views, n_views)
                            except:
                                print('error of slide mask:', slide_mask)
                                exit(1)
                            slide_mask[i, start:end] = 1
                        # print('slide_mask:', slide_mask)
                        src_indices = full_indices[(~(full_indices == cur_indices[:,None]))*slide_mask].view(1,n_views,min(n_views, num_context_views)-1).repeat(b,1,1)
                    src_extrinsics = context['extrinsics'][:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,4,4))
                    src_intrinsics = context_intrinsics[:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,3,3))
                    src_image = context['image'][:,None].repeat(1,n_views,1,1,1,1).gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,3,h,w))\
                                            .view(-1,n_views-1 if not judge else min(n_views, num_context_views)-1,3,h,w)
                    src_cam_t_world = src_extrinsics.inverse()
                    cur_cam_t_world = cur_extrinsics.inverse()
                    src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics.unsqueeze(2)
                    cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics
                    matching_cur_feats, matching_src_feats = self.compute_matching_feats(
                                                            cur_image, src_image, is_testing)
                    
                    src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
                    cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
                    src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
                    cur_intrinsics_ = rearrange(cur_intrinsics, 'b v x y -> (b v) x y')
                    src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
                    src_K[:,:,:3,:3] = src_intrinsics_
                    cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
                    cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()
                    
                    # print('near.shape:', context["near"].shape)
                    # min_depth = torch.tensor(0.5).type_as(src_K).view(1, 1, 1, 1)
                    # max_depth = torch.tensor(15.0).type_as(src_K).view(1, 1, 1, 1)
                    # print('min_depth.shape:', min_depth.shape)

                    near = context["near"][:1,0].type_as(src_K).view(1, 1, 1, 1)
                    far = context["far"][:1,0].type_as(src_K).view(1, 1, 1, 1)

                    # print('cv depth range:', near, far)

                    cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(
                                                        cur_feats=matching_cur_feats,
                                                        src_feats=matching_src_feats,
                                                        src_extrinsics=src_cam_T_cur_cam_,
                                                        src_poses=cur_cam_T_src_cam_,
                                                        src_Ks=src_K,
                                                        cur_invK=cur_inverse,
                                                        min_depth=near,
                                                        max_depth=far,
                                                    )
                    
                    cost_volume_features = self.cv_encoder(
                                            cost_volume, 
                                            cur_feats[1:],
                                        )
                    # print('cost_volume:', cost_volume)
                    # print('cost_volume_features:', cost_volume_features)
                    cur_feats = cur_feats[:1] + cost_volume_features


                depth_outputs = self.depth_decoder(cur_feats)
                # for k in depth_outputs.keys():
                #     try:
                #         print(f'{k}.shape:', depth_outputs[k].shape)
                #     except:
                #         pass
                # exit(0)
                # print('keys:', depth_outputs.keys())
                # print('depth_pred.shape:', depth_outputs[f'depth_plane_pred'].shape)

                # for k in list(depth_outputs.keys()):
                #     log_depth = depth_outputs[k][:,:1].float()
                #     depth_outputs[k.replace('output', 'depth')] = log_depth
                #     depth_outputs[k.replace("log_output", "depth")] = torch.exp(log_depth)
                #     raw_opacity = depth_outputs[k][:,1:2].float()
                #     depth_outputs[k.replace("log_output", "density")] = nn.Sigmoid()(raw_opacity)
                
                to_skip = context['image']
                to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
                # print('post keys:', depth_outputs.keys())
                # print('post depth_pred.shape:', depth_outputs[f'depth_plane_pred'].shape)
                
                
                for s in range(-1, self.cfg.n_levels+1):
                
                    # depths = depth_outputs[f'depth_pred_s{s}_b1hw'].view(b,n_views,-1,1,1)
                    # depths = depth_outputs[f'depth_plane_pred'].view(b,n_views,-1,1,1)
                    
                    
                    # print('USING GT DEPTH !!!!!!!')
                    # depths = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                    # depths[(depths==0) | (depths>15)] = 100
                    
                    
                    # print('depths.shape:', depths.shape)
                    
                    # print('gaussians_feats.shape:', gaussians_feats.shape)
                    
                    skip = self.high_resolution_skip[s+1](to_skip)

                    # print('skip.shape:', skip.shape)

                    
                    

                    if not export_ply:
                        xy_ray, _ = sample_image_grid((h, w), device)
                    else:
                        if not resized:
                            margin = 4
                            context["intrinsics"] = context["intrinsics"] * torch.tensor([[w*1.0 / (w - 4), h*1.0 / (h - 4), 1]], device=device)
                            h = h - margin*2
                            w = w - margin*2
                            context['image_shape'] = (context['image_shape'][0]-margin*2, context['image_shape'][1]-8)
                            resized = 1
                        xy_ray, _ = sample_image_grid((h, w), device)
                        xy_ray = xy_ray + torch.tensor([[[2]]], dtype=torch.float32, device=device)
                        depth_outputs[f'output_pred_s{s}_b1hw'] = depth_outputs[f'output_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                        depth_outputs[f'depth_pred_s{s}_b1hw'] = depth_outputs[f'depth_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                        depth_outputs[f'depth_weights'] = depth_outputs[f'depth_weights'][:, :, margin:-margin, margin:-margin]
                        skip = skip[:, :, margin:-margin, margin:-margin]
                    # print('h, w:', h, w)
                    # print('output shape:', depth_outputs[f'output_pred_s{s}_b1hw'].shape)

                    gaussians_feats = rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views)#.view(b,n_views,h//(2**(s+1)), w//(2**(s+1)),64)
                    # gaussians_feats = depth_outputs[f'output_pred_s{s}_b1hw'][:,1:].view(b,n_views,h//(2**(s+1)), w//(2**(s+1)),64)
                    gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=n_views)
                    # densities = nn.Sigmoid()(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1].view(b,n_views,-1,1,1))
                    densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views))
                    depths = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
                    weights = rearrange(depth_outputs[f'depth_weights'], "(b v) c h w -> b v (c h w) () ()", b=b)
                    
                    gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c")
                    xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                    offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                                                    device=gaussians_feats.device)
                    xy_ray = xy_ray + offset_xy

                    if not self.cfg.fusion:
                        gaussians_now = rearrange(
                            self.to_gaussians(gaussians_feats),
                            "... (srf c) -> ... srf c",
                            srf=self.cfg.num_surfaces,
                        )
                        xy_ray, _ = sample_image_grid((h//(2**(s+1)), w//(2**(s+1))), device)
                        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                        offset_xy = gaussians_now[..., :2].sigmoid()
                        # print('xy_ray.shape:', xy_ray.shape) # [196608, 1, 2]
                        # print('offset_xy.shape:', offset_xy.shape) # [1, 2, 49152, 1, 2]
                        pixel_size = 1 / torch.tensor((w//(2**(s+1)), h//(2**(s+1))), dtype=torch.float32, device=device)
                        # xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                        xy_ray = xy_ray + torch.zeros_like(offset_xy, device=offset_xy.device)

                        gpp = self.cfg.gaussians_per_pixel


                        # depths = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")

                        # print('(h,w):', (h,w))
                        # exit(0)
                        # print(f'densities.shape:', densities.shape)
                        # print(f'weights.shape:', weights.shape)
                        gaussians_now = self.gaussian_adapter.forward(
                            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                            depths,
                            # self.map_pdf_to_opacity(densities, global_step) / gpp,
                            # torch.clip(densities / (2**s), 0, 1),
                            # densities,
                            nn.Sigmoid()(gaussians_now[..., :1]),
                            # torch.ones(densities.shape, device=densities.device),
                            # weights,
                            rearrange(gaussians_now[..., 2:], "b v r srf c -> b v r srf () c"),
                            (h//(2**(s+1)), w//(2**(s+1))),
                            load_depth=self.cfg.load_depth,
                            fusion=(num>0) and (num!=length//2),
                        )
                        
                        gaussians.append(gaussians_now)
                    else:
                        coords.append(self.gaussian_adapter.forward(
                            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                            depths,
                            densities,
                            gaussians_feats,
                            (h//(2**(s+1)), w//(2**(s+1))),
                            load_depth=self.cfg.load_depth,
                            fusion=True,
                        ))
                        gaussians.append(gaussians_feats)
                    # try:
                    results[f'depth_num{num}_s{s}'] = depths
                    try:
                        depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                        results[f'depth_num{num}_s{s}_raw'] = depths_raw
                        mask = (depths_raw > 1e-3) * (depths_raw < 10)
                        results[f'depth_num{num}_s{s}_mask'] = mask
                    except:
                        pass
                    results[f'depth_num{num}_s{s}_b1hw'] = depth_outputs[f'depth_pred_s{s}_b1hw']

                    for s in range(4):
                        results[f'log_depth_num{num}_s{s}'] = rearrange(depth_outputs[f'log_depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
                        results[f'depth_num{num}_s{s}'] = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)

                        log_depths = depth_outputs[f'log_depth_pred_s{s}_b1hw']
                        depths = depth_outputs[f'depth_pred_s{s}_b1hw']
                        try:
                            # depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                            depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
                            results[f'depth_num{num}_s{s}_raw_b1hw'] = depths_raw
                            mask = (depths_raw > 1e-3) * (depths_raw < 10)
                            results[f'depth_num{num}_s{s}_mask_b1hw'] = mask
                        except:
                            pass
                        
                        # print('USING GT DEPTH !!!!!!!')
                        # depths = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                        results[f'depth_num{num}_s{s}_b1hw'] = depths
                        results[f'log_depth_num{num}_s{s}_b1hw'] = log_depths
                        # depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
                # results[f'coarse_depth_pred'] = depth_outputs[f'coarse_depth_pred'].view(b,n_views,-1,1,1)
                # s = 0
                # depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                # results[f'depth_s{s}_raw'] = depths_raw
                # mask = (depths_raw > 0) * (depths_raw < 15)
                # results[f'depth_s{s}_mask'] = mask
                # except:
                #     pass
                
                # print('results.keys():', results.keys())
                
                # exit(0)

                # for k in results.keys():
                #     if 'depth' in k:
                #         print(f'{k}.shape:', results[k].shape)
                # depths = depth_outputs['depth_pred_s0_b1hw'].view(b,n_views,-1,1,1)
                            
            our_gaussians = []

            if not self.cfg.fusion:
            # if True:
                num_gaussians = gaussians[0].means.shape[2] * gaussians[0].means.shape[1]
                gaussians = G(means=torch.cat([rearrange(x.means, "b v r srf spp c -> b () (v r) srf spp c") for x in gaussians], dim=2),
                                covariances=torch.cat([rearrange(x.covariances, "b v r srf spp c d -> b () (v r) srf spp c d") for x in gaussians], dim=2),
                                scales=torch.cat([rearrange(x.scales, "b v r srf spp c -> b () (v r) srf spp c") for x in gaussians], dim=2),
                                rotations=torch.cat([rearrange(x.rotations, "b v r srf spp c -> b () (v r) srf spp c") for x in gaussians], dim=2),
                                harmonics=torch.cat([rearrange(x.harmonics, "b v r srf spp c d -> b () (v r) srf spp c d") for x in gaussians], dim=2),
                                opacities=torch.cat([rearrange(x.opacities, "b v r srf spp -> b () (v r) srf spp") for x in gaussians], dim=2),
                                )
                our_gaussians.append(gaussians)
            else:
                num_raw_gaussians = gaussians[0].shape[2] * gaussians[0].shape[1]
                # weight_emb = self.weight_embedding(torch.cat([weights, densities], dim=-1))
                B = gaussians[0].shape[0]
                for b in range(B):
                    cur_gs = [x[b:b+1] for x in gaussians]
                    cur_coords = [x[b:b+1] for x in coords]
                    cur_densities = densities[b:b+1]
                    cur_weights = weights[b:b+1]
                    # print('depth_outputs[f\'depth_pred_s-1_b1hw\'].shape:', depth_outputs[f'depth_pred_s-1_b1hw'].shape, 'B:', B)
                    cur_depth = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v c h w", b=B)[b]
                    cur_gaussians, cur_coords, cur_extrinsics, cur_depths = self.fuse_gaussians(cur_gs, cur_coords, 
                                                    cur_densities, cur_weights, 
                                                    cur_depth, 
                                                    context["extrinsics"][b:b+1], \
                                                    context["intrinsics"][b:b+1], context['image_shape'])
                    # gaussians, coords, extrinsics, depths = self.fuse_gaussians(gaussians, coords, densities, weights, depth_outputs['depth_pred_s-1_b1hw'], 
                    #                                 context["extrinsics"], \
                    #                                 context["intrinsics"], context['image_shape'])
                    cur_gaussians_now = rearrange(
                                    self.to_gaussians(cur_gaussians),
                                    "... (srf c) -> ... srf c",
                                    srf=self.cfg.num_surfaces,
                                )
                    # offset_xy = gaussians_now[..., :2].sigmoid()
                    # print('xy_ray.shape:', xy_ray.shape) # [196608, 1, 2]
                    # print('offset_xy.shape:', offset_xy.shape) # [1, 2, 49152, 1, 2]
                    # pixel_size = 1 / torch.tensor((w//(2**(s+1)), h//(2**(s+1))), dtype=torch.float32, device=device)
                    # xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                    # xy_ray = xy_ray + torch.zeros_like(offset_xy, device=offset_xy.device)

                    # gpp = self.cfg.gaussians_per_pixel


                    # depths = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")

                    # print('(h,w):', (h,w))
                    # exit(0)
                    # print(f'densities.shape:', densities.shape)
                    # print(f'weights.shape:', weights.shape)
                    cur_gaussians = self.gaussian_adapter.forward(
                        rearrange(cur_extrinsics, "b r i j -> b () r () () i j"),
                        repeat(context["intrinsics"][b:b+1,0], "b i j -> b () N () () i j", N=cur_gaussians_now.shape[1]),
                        rearrange(xy_ray[b:b+1], "b v r srf xy -> b v r srf () xy"),
                        rearrange(cur_depths, "b r -> b () r () ()"),
                        # self.map_pdf_to_opacity(densities, global_step) / gpp,
                        # torch.clip(densities / (2**s), 0, 1),
                        # rearrange(densities, "b r srf c -> b () r srf c"),
                        nn.Sigmoid()(rearrange(cur_gaussians_now[..., :1], "b r srf c -> b () r srf c")),
                        # torch.ones(densities.shape, device=densities.device),
                        # weights,
                        rearrange(cur_gaussians_now[..., 2:], "b r srf c -> b () r srf () c"),
                        (h, w),
                        load_depth=self.cfg.load_depth,
                        fusion=False,
                        coords=rearrange(cur_coords, "b r c -> b () r () () c"),
                    )
                    our_gaussians.append(cur_gaussians)
                num_gaussians = our_gaussians[0].means.shape[2]
                results['gs_ratio'] = num_gaussians / num_raw_gaussians
                gaussians = cur_gaussians
        
        
        
        
        results['num_gaussians'] = num_gaussians
            # print('depths.shape:', depths.shape)
            # print('gaussians.shape:', gaussians.shape)
            


            

            # depths = depths_raw

            # print('depths_raw.shape:', depths_raw.shape)
            # print('depths.shape:', depths.shape)
            # print('pre_depths_raw.shape:', context['depth'].shape)
            # depths_raw = mmcv.imresize(rearrange(context['depth'], "b v c h w -> (b v) c h w",\
            #                                      b=b, v=n_views).cpu().numpy(), (w,h), interpolation='nearest')
            # depths_raw = torch.tensor(depths_raw, device=depths.device)
            # print('post_depths_raw.shape:', depths_raw.shape)
            # depths_raw = rearrange(depths_raw, "(b v) c h w -> b v (h w) c 1", b=b, v=n_views)

            # results = {}
            # results['depths'] = depths
            # results['depths_raw'] = depths_raw
            # mask = (depths_raw > 0) * (depths_raw < 15)
            # results['depths_mask'] = mask
            # print('depth_outputs:', depth_outputs)

            # print('using gt depths')
            # print('depths.shape:', depths.shape)
        # print('depths.shape:', depths.shape)
        # print('densities.shape:', densities.shape)
        # print('gt_depth.shape:', context['depth'].shape)
        # torch.save({'gt_depth': context['depth'], 'depths': depths}, 'test.th')
        # exit(0)
        # print('depths.shape:', depths.shape)

        # Convert the features and depths into Gaussians.
        
        

        # Dump visualizations if needed.
        # if visualization_dump is not None:
        if True:
            visualization_dump = {}
            # visualization_dump["depth"] = rearrange(
            #     depths, "b v (h w) srf s -> b v h w srf s", h=h//2**(self.cfg.est_depth == 'cost'), w=w//2**(self.cfg.est_depth == 'cost')
            # )
            try:
                visualization_dump["scales"] = rearrange(
                    gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                )
                visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )
            except:
                visualization_dump["scales"] = gaussians.scales
                visualization_dump["rotations"] = gaussians.rotations
            
            # if self.cfg.use_epipolar_transformer and (self.load_depth is None):
            if self.cfg.use_epipolar_transformer:
                visualization_dump["sampling"] = sampling
            
            results['visualizations'] = visualization_dump

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = (
            rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
            if self.cfg.predict_opacity
            else 1
        )
        if not self.cfg.fusion:
            results['gaussians'] =  Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    opacity_multiplier * gaussians.opacities if not self.cfg.op1 else
                    torch.ones_like(gaussians.opacities, device=gaussians.opacities.device),
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
        else:
            del gaussians
            final_gs = []
            # print('len(our_gaussians):', len(our_gaussians))
            # print('our_gaussians[0].covariances.shape:', our_gaussians[0].covariances.shape)
            for i in range(len(our_gaussians)):
                final_gs.append(Gaussians(
                    rearrange(
                        our_gaussians[i].means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    rearrange(
                        our_gaussians[i].covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    rearrange(
                        our_gaussians[i].harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    rearrange(
                        opacity_multiplier * our_gaussians[i].opacities if not self.cfg.op1 else
                        torch.ones_like(our_gaussians[i].opacities, device=our_gaussians[i].opacities.device),
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                ))
            results['gaussians'] = final_gs
        return results

    def fuse_gaussians(self, gaussians, coords, densities, weight_emb, depths, 
                       extrinsics, intrinsics, image_shape, depth_thres=0.1, limit=100):
        # print('gaussians[0].means.shape:', gaussians[0].means.shape) # [1, 3, 196608, 1, 1, 3]
        # print('extrinsics.shape:', extrinsics.shape) # [1, 3, 4, 4]
        # print('intrinsics.shape:', intrinsics.shape) # [1, 3, 3, 3]
        # exit(0)
        # print('gaussians[0].shape:', gaussians[0].shape)
        # print('coords[0].shape:', coords[0].shape)
        # print('densities.shape:', densities.shape)
        length = min(gaussians[0].shape[1], limit)
        global_gaussians = gaussians[0][:,0]
        global_densities = densities[:, 0]
        global_weight_emb = weight_emb[:, 0]
        global_coords = coords[0][:,0,:,0,0]
        global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians.shape[1],1,1)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        global_depths = depths[None, 0]
        # print('depths.shape:', depths.shape)
        h, w = image_shape
        # print('h,w:', h,w)
        for i in range(1, length):
        # for i in range(1,2):
        # for i in range(0, 1):
            extrinsic = extrinsics[0,i]
            intrinsic = intrinsics[0,i].clone()
            intrinsic[:1,:] *= w
            intrinsic[1:2,:] *= h
            focal_length = (intrinsic[0, 0], intrinsic[1, 1])
            principal_point = (intrinsic[0, 2], intrinsic[1, 2])
            principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
            principal_point_mat = principal_point_mat.reshape(1, 2)
            focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
            focal_length_mat = focal_length_mat.reshape(1, 2)
            means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
            post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
            curr_depths = post_xy_coords[2:3, :]
            post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
            curr_depths = curr_depths.squeeze()
            # print('focal_length_mat:', focal_length_mat)
            # print('principal_point_mat:', principal_point_mat)
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype)
            depth_map = torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype) * 10000
            # for index in valid_indices:
            #     x, y = pixel_coords[index]
            #     if (x >= 0) and (x < w) and (y >= 0) and (y < h) and (curr_depths[index] > 0) and (curr_depths[index] < 15):
            #         if curr_depths[index] < depth_map[y, x]:
            #             depth_map[y, x] = curr_depths[index]
            #             proj_map[y, x] = index

            # record the minimum depth for each pixel and the corresponding point index 
            # print('pixel_coords:', pixel_coords)
            # print('valid.sum:', valid.sum())
            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            # print('pixel_indices:', pixel_indices)
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')
            # plt.imsave('depth_map.png', depth_map.cpu().numpy().reshape(h,w))
            # plt.imsave('depths.png', depths[i].cpu().numpy().reshape(h,w))


            # fusion_mask = torch.abs(depth_map - depths[i]) < depth_thres
            fusion_mask = torch.abs(depth_map - depths[i]) < depths[i] * 0.05

            # plt.imsave('fusion_mask.png', fusion_mask.cpu().numpy().reshape(h,w))
            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            # print('proj_map.shape:', proj_map.shape)
            # print('valid.sum:', valid.sum())
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            # print('fusion_indices_.sum:', fusion_indices_.sum())
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices
            # plt.imsave('mask.png', mask.cpu().numpy().reshape(h,w))
            # plt.imsave('curr_depths.png', curr_depths.cpu().numpy().reshape(h,w))
            # exit(0)
            # print('global_means.shape:', global_gaussians.means.shape)
            # weights_0 = torch.ones_like(global_gaussians.opacities[:, mask].unsqueeze(-1), device=global_gaussians.opacities.device)
            # weights_1 = torch.ones_like(gaussians[0].opacities[:, i, pixel_indices][:,fusion_indices_].unsqueeze(-1), device=global_gaussians.opacities.device)
            # weights_0 = global_densities[:, mask].squeeze(-1)
            # weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].squeeze(-1)
            # print('weights_0.shape:', weights_0.shape)
            # print('weights_1.shape:', weights_1.shape)
            # print('global_gaussians.shape:', global_gaussians.shape)
            # print('global_coords.shape:', global_coords.shape)
            # print('coords[0].shape:', coords[0].shape)
            # print('global_depths.shape:', global_depths.shape)
            # print('depths.shape:', depths.shape)
            # print('global_gaussians[:, mask].shape:', global_gaussians[:, mask].shape)
            # print('gaussians[0][:, i, pixel_indices][:,fusion_indices_].shape:', gaussians[0][:, i, pixel_indices][:,fusion_indices_].shape)
            # exit(0)
            # mask = torch.zeros_like(mask, device=mask.device, dtype=torch.bool)
            if mask.sum() > 0:
                input_weights_emb = positional_encoding(torch.cat([global_densities[:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], global_weight_emb[:, mask]], dim=-1), 6)
                fusion_feat = self.gru(gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2),
                                       global_gaussians[:, mask].unsqueeze(2),
                                       input_weights_emb,
                                       hidden_weights_emb).squeeze(2)

                global_gaussians = torch.cat([global_gaussians[:, ~mask], fusion_feat], dim=1)
                # fusion_feat = torch.cat([global_gaussians[:, mask].unsqueeze(2), gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2)], dim=2)
                # fusion_feat = rearrange(fusion_feat, "b N V C -> (b N) V C")
                # fusion_feat = self.transformer(fusion_feat)
                # fusion_feat = F.softmax(self.blending(fusion_feat), dim=-1)
                # # print('fusion_feat.shape:', fusion_feat.shape)
                # fusion_feat = rearrange(fusion_feat, "(b N) V C -> b N V C", b=1)
                # weights_0, weights_1 = torch.split(fusion_feat, 1, dim=2)
                # # else:
                weights_0 = global_densities[:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)
                # global_gaussians = torch.cat([global_gaussians[:, ~mask], (global_gaussians[:, mask]*weights_0[...,0] +
                #                                 gaussians[0][:, i, pixel_indices][:,fusion_indices_]*weights_1[...,0]) / (weights_0[...,0]+weights_1[...,0])], dim=1)
                global_coords = torch.cat([global_coords[:, ~mask], (global_coords[:, mask]*weights_0[...,1] +
                                                coords[0][:, i, pixel_indices][:,fusion_indices_,0,0]*weights_1[...,1]) / (weights_0[...,1]+weights_1[...,1])], dim=1)
                # # global_gaussians.covariances = torch.cat([global_gaussians.covariances[:, ~mask], (global_gaussians.covariances[:, mask]*weights_0.unsqueeze(-1) +
                # #                                         gaussians[0].covariances[:, i, pixel_indices][:,fusion_indices_]*weights_1.unsqueeze(-1)) / (weights_0.unsqueeze(-1)+weights_1.unsqueeze(-1))], dim=1)
                # # global_gaussians.scales = torch.cat([global_gaussians.scales[:, ~mask], (global_gaussians.scales[:, mask]*weights_0 +
                # #                                         gaussians[0].scales[:, i, pixel_indices][:,fusion_indices_]*weights_1) / (weights_0+weights_1)], dim=1)
                # # global_gaussians.rotations = torch.cat([global_gaussians.rotations[:, ~mask], (global_gaussians.rotations[:, mask]*weights_0 +
                # #                                         gaussians[0].rotations[:, i, pixel_indices][:,fusion_indices_]*weights_1) / (weights_0+weights_1)], dim=1)
                # # global_gaussians.harmonics = torch.cat([global_gaussians.harmonics[:, ~mask], (global_gaussians.harmonics[:, mask]*weights_0.unsqueeze(-1) +
                # #                                         gaussians[0].harmonics[:, i, pixel_indices][:,fusion_indices_]*weights_1.unsqueeze(-1)) / (weights_0.unsqueeze(-1)+weights_1.unsqueeze(-1))], dim=1)
                global_densities = torch.cat([global_densities[:, ~mask], (global_densities[:, mask] +
                                                        densities[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                global_weight_emb = torch.cat([global_weight_emb[:, ~mask], (global_weight_emb[:, mask] +
                                                        weight_emb[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                
                global_extrinsics = torch.cat([global_extrinsics[:, ~mask], (global_extrinsics[:, mask]*weights_0[...,:1] +
                                                extrinsics[:, i, None]*weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1])], dim=1)
                global_depths = torch.cat([global_depths[:, ~mask], (global_depths[:, mask]*weights_0[...,0,0] +
                                                        depths[None, i, pixel_indices][:,fusion_indices_]*weights_1[...,0,0])/(weights_0[...,0,0]+weights_1[...,0,0])], dim=1)
            
            
            
            # global_gaussians.means = torch.cat([global_gaussians.means[:, ~mask], (global_gaussians.means[:, mask] +
            #                                         gaussians[0].means[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)
            # global_gaussians.covariances = torch.cat([global_gaussians.covariances[:, ~mask], (global_gaussians.covariances[:, mask] +
            #                                         gaussians[0].covariances[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)
            # global_gaussians.scales = torch.cat([global_gaussians.scales[:, ~mask], (global_gaussians.scales[:, mask] +
            #                                         gaussians[0].scales[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)
            # global_gaussians.rotations = torch.cat([global_gaussians.rotations[:, ~mask], (global_gaussians.rotations[:, mask] +
            #                                         gaussians[0].rotations[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)
            # global_gaussians.harmonics = torch.cat([global_gaussians.harmonics[:, ~mask], (global_gaussians.harmonics[:, mask] +
            #                                         gaussians[0].harmonics[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)
            # global_gaussians.opacities = torch.cat([global_gaussians.opacities[:, ~mask], (global_gaussians.opacities[:, mask] +
            #                                         gaussians[0].opacities[:, i, pixel_indices][:,fusion_indices_]) / 2], dim=1)


            # global_gaussians.means = gaussians[0].means[:, i, fusion_mask]
            # global_gaussians.covariances = gaussians[0].covariances[:, i, fusion_mask]
            # global_gaussians.scales = gaussians[0].scales[:, i, fusion_mask]
            # global_gaussians.rotations = gaussians[0].rotations[:, i, fusion_mask]
            # global_gaussians.harmonics = gaussians[0].harmonics[:, i, fusion_mask]
            # global_gaussians.opacities = gaussians[0].opacities[:, i, fusion_mask]
            global_gaussians = torch.cat([global_gaussians, gaussians[0][:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_coords = torch.cat([global_coords, coords[0][:,i]\
                                                [:,~fusion_mask,0,0]], dim=1)
            # global_gaussians.covariances = torch.cat([global_gaussians.covariances, gaussians[0].covariances[:,i]\
            #                                     [:,~fusion_mask]], dim=1)
            # global_gaussians.scales = torch.cat([global_gaussians.scales, gaussians[0].scales[:,i]\
            #                                     [:,~fusion_mask]], dim=1)
            # global_gaussians.rotations = torch.cat([global_gaussians.rotations, gaussians[0].rotations[:,i]\
            #                                     [:,~fusion_mask]], dim=1)
            # global_gaussians.harmonics = torch.cat([global_gaussians.harmonics, gaussians[0].harmonics[:,i]\
            #                                     [:,~fusion_mask]], dim=1)
            global_densities = torch.cat([global_densities, densities[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_weight_emb = torch.cat([global_weight_emb, weight_emb[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_extrinsics = torch.cat([global_extrinsics, extrinsics[:,i,None].repeat(1,(~fusion_mask).sum(),1,1)], dim=1)
            global_depths = torch.cat([global_depths, depths[None,i]\
                                                [:,~fusion_mask]], dim=1)
            # exit(0)
            # print('pixel_coords:', pixel_coords)
            # print('fusion_mask:', fusion_mask)
            # plt.imsave('fusion_map.png', fusion_map.cpu().numpy())
            # print('post_xy_coords:', post_xy_coords.reshape(-1,2))

        return global_gaussians, global_coords, global_extrinsics, global_depths

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.epipolar_transformer.self_attention.patch_size
                * self.cfg.epipolar_transformer.downscale,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler
