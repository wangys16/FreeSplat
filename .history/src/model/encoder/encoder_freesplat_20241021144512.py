from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .modules.networks import CVEncoder, DepthDecoder
from .modules.cost_volume import AVGFeatureVolumeManager

import timm
from .modules.layers import TensorFormatter

from .modules.networks import GRU

from einops import *


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


UseDepthMode = Literal[
    "depth"
]

def rotation_distance(rotations):
    R1 = rotations.unsqueeze(2) 
    R2 = rotations.unsqueeze(1) 
    R_rel = torch.matmul(R1.transpose(-2, -1), R2) 

    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) 
    trace = torch.clamp(trace, -1, 3)
    angle = torch.acos((trace - 1) / 2)
    return angle.squeeze(0) 

def calculate_distance_matrix(poses):
    translations = poses[:, :, :3, 3]
    rotations = poses[:, :, :3, :3]
    
    translation_dist = torch.cdist(translations, translations).squeeze(0)
    
    rotation_dist = rotation_distance(rotations)
    
    combined_dist = translation_dist + rotation_dist

    return combined_dist

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


def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()

@dataclass
class EncoderFreeSplatCfg:
    name: Literal["freesplat"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    use_transmittance: bool
    
    num_depth_candidates: int = 64
    num_views: int = 2
    image_H: int = 384
    image_W: int = 512

class EncoderFreeSplat(Encoder[EncoderFreeSplatCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderFreeSplatCfg, depth_range=[0.5, 15.0]) -> None:
        super().__init__(cfg)
        activation_func = nn.ReLU()

        self.depth_range = depth_range

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.backbone = timm.create_model(
                                        "tf_efficientnetv2_s_in21ft1k", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        self.backbone.num_ch_enc = self.backbone.feature_info.channels()
        
        self.high_resolution_skip = nn.ModuleList(
                                        [nn.Sequential(
                                            nn.Conv2d(3, 64, 7, 1, 3),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 6, 2, 2),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 8, 4, 2),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 16, 8, 4),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 32, 16, 8),
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
        
        self.cost_volume = AVGFeatureVolumeManager(matching_height=self.cfg.image_H//4, 
                                                    matching_width=self.cfg.image_W//4,
                                                    num_depth_bins=self.cfg.num_depth_candidates,
                                                    matching_dim_size=48,)
        self.cv_encoder = CVEncoder(num_ch_cv=self.cfg.num_depth_candidates,
                                    num_ch_enc=self.backbone.num_ch_enc[1:],
                                    num_ch_outs=[64, 128, 256, 384])
        dec_num_input_ch = (self.backbone.num_ch_enc[:1] 
                                        + self.cv_encoder.num_ch_enc)

        self.depth_decoder = DepthDecoder(dec_num_input_ch, 
                                            num_output_channels=1+64,
                                            near=depth_range[0],
                                            far=depth_range[1],
                                            num_samples=self.cfg.num_depth_candidates,)
        self.max_depth = 4
        self.tensor_formatter = TensorFormatter()

        self.weight_embedding = nn.Sequential(nn.Linear(2, 12), 
                                    activation_func,
                                    nn.Linear(12, 12),)
        self.gru = GRU()

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
    
    def forward(
        self,
        context,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        is_testing: bool = False,
        export_ply: bool = False,
        dataset_name: str = 'scannet',
    ) -> dict:
        device = context["image"].device
        b, n_views, _, h, w = context["image"].shape
        results = {}
        num_context_views = self.cfg.num_views

        gaussians = []
        coords = []
        results = {}

        # Apply training mode only to batch normalization layers
        self.backbone.apply(set_bn_eval)
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

        full_indices = torch.arange(n_views, device=context['image'].device)[None].repeat(n_views,1)
        
        use_local = (n_views > num_context_views)
        if not use_local:
            src_indices = full_indices[~(full_indices == cur_indices[:,None])].view(1,n_views,n_views-1).repeat(b,1,1)
        else:
            slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
            dist_matrix = calculate_distance_matrix(context["extrinsics"])

            # For each row in the distance matrix, mark the closest 'num_context_views' entries as True
            _, indices = torch.topk(dist_matrix, min(num_context_views, n_views), largest=False, sorted=False, dim=1)
            slide_mask.scatter_(1, indices, True)
            slide_mask[torch.arange(n_views), torch.arange(n_views)] = False

            src_indices = full_indices[(~(full_indices == cur_indices[:,None]))*slide_mask].view(1,n_views,min(n_views, num_context_views)-1).repeat(b,1,1)

        
        src_extrinsics = context['extrinsics'][:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,4,4))
        src_intrinsics = context_intrinsics[:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,3,3))
        src_image = context['image'][:,None].repeat(1,n_views,1,1,1,1).gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,3,h,w))\
                                .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,3,h,w)
        src_cam_t_world = src_extrinsics.inverse()
        cur_cam_t_world = cur_extrinsics.inverse()
        src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics.unsqueeze(2)
        cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics

        matching_cur_feats = cur_feats[1]
        dim = matching_cur_feats.shape[-3]
        matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=n_views)[:,None].repeat(1,n_views,1,1,1,1).\
                            gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,dim,h//4,w//4))\
                            .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,dim,h//4,w//4)

        src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
        cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
        src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
        cur_intrinsics_ = rearrange(cur_intrinsics, 'b v x y -> (b v) x y')
        src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
        src_K[:,:,:3,:3] = src_intrinsics_
        cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
        cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()
        

        near = context["near"][:1,0].type_as(src_K).view(1, 1, 1, 1)
        far = context["far"][:1,0].type_as(src_K).view(1, 1, 1, 1)


        cost_volume = self.cost_volume(cur_feats=matching_cur_feats,
                                        src_feats=matching_src_feats,
                                        src_extrinsics=src_cam_T_cur_cam_,
                                        src_poses=cur_cam_T_src_cam_,
                                        src_Ks=src_K,
                                        cur_invK=cur_inverse,
                                        min_depth=near,
                                        max_depth=far,
                                        context_images=None,
                                    )

        cost_volume_features = self.cv_encoder(
                                cost_volume, 
                                cur_feats[1:],
                            )
        cur_feats = cur_feats[:1] + cost_volume_features
            
        depth_outputs = self.depth_decoder(cur_feats)
        
        to_skip = context['image']
        to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")

        skip = self.high_resolution_skip[0](to_skip)
        xy_ray, _ = sample_image_grid((h, w), device)
        gaussians_feats = rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views)
        gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=n_views)
        densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views))
        depths = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
        weights = rearrange(depth_outputs[f'depth_weights'], "(b v) c h w -> b v (c h w) () ()", b=b)
        
        gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c")
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                                        device=gaussians_feats.device)
        xy_ray = xy_ray + offset_xy

        coords.append(self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                densities,
                gaussians_feats,
                (h, w),
                fusion=True,
            ))
        gaussians.append(gaussians_feats)

        results[f'depth_num0_s-1'] = depths
        try:
            depths_raw = rearrange(context[f'depth_s-1'], "b v c h w -> b v (h w) c 1")
            results[f'depth_num0_s-1_raw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num0_s-1_mask'] = mask
        except:
            pass
        results[f'depth_num0_s-1_b1hw'] = depth_outputs[f'depth_pred_s-1_b1hw']

        for s in range(self.max_depth):
            results[f'depth_num0_s{s}'] = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)

            log_depths = depth_outputs[f'log_depth_pred_s{s}_b1hw']
            depths = depth_outputs[f'depth_pred_s{s}_b1hw']
            try:
                depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
                results[f'depth_num0_s{s}_raw_b1hw'] = depths_raw
                mask = (depths_raw > 1e-3) * (depths_raw < 10)
                results[f'depth_num0_s{s}_mask_b1hw'] = mask
            except:
                pass
            
            results[f'depth_num0_s{s}_b1hw'] = depths
                        
        our_gaussians = []
        num_raw_gaussians = gaussians[0].shape[2] * gaussians[0].shape[1]
        B = gaussians[0].shape[0]
        for b in range(B):
            cur_gs = [x[b:b+1] for x in gaussians]
            cur_coords = [x[b:b+1] for x in coords]
            cur_densities = densities[b:b+1]
            cur_weights = weights[b:b+1]
            cur_depth = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v c h w", b=B)[b]
            cur_gaussians, cur_coords, cur_extrinsics, cur_depths = self.fuse_gaussians(cur_gs, cur_coords, 
                                            cur_densities, cur_weights, 
                                            cur_depth, 
                                            context["extrinsics"][b:b+1], \
                                            context["intrinsics"][b:b+1], context['image_shape'])

            cur_gaussians_now = rearrange(
                            self.to_gaussians(cur_gaussians),
                            "... (srf c) -> ... srf c",
                            srf=self.cfg.num_surfaces,
                        )
            cur_gaussians = self.gaussian_adapter.forward(
                rearrange(cur_extrinsics, "b r i j -> b () r () () i j"),
                repeat(context["intrinsics"][b:b+1,0], "b i j -> b () N () () i j", N=cur_gaussians_now.shape[1]),
                rearrange(xy_ray[b:b+1], "b v r srf xy -> b v r srf () xy"),
                rearrange(cur_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(cur_gaussians_now[..., :1], "b r srf c -> b () r srf c")),
                rearrange(cur_gaussians_now[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                fusion=False,
                coords=rearrange(cur_coords, "b r c -> b () r () () c"),
            )
            our_gaussians.append(cur_gaussians)
        num_gaussians = our_gaussians[0].means.shape[2]
        results['gs_ratio'] = num_gaussians / num_raw_gaussians
        gaussians = cur_gaussians
 
        results['num_gaussians'] = num_gaussians
        visualization_dump = {}
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
        
        results['visualizations'] = visualization_dump

        final_gs = []
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
                    our_gaussians[i].opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            ))
        results['gaussians'] = final_gs

        return results

    def fuse_gaussians(self, gaussians, coords, densities, weight_emb, depths, 
                       extrinsics, intrinsics, image_shape, depth_thres=0.1):
        length = gaussians[0].shape[1]
        global_gaussians = gaussians[0][:,0]
        global_densities = densities[:, 0]
        global_weight_emb = weight_emb[:, 0]
        global_coords = coords[0][:,0,:,0,0]
        global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians.shape[1],1,1)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        global_depths = depths[None, 0]

        h, w = image_shape
        for i in range(1, length):
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
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype)
            depth_map = torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype) * 10000

            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

            fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres)

            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            if mask.sum() > 0:
                input_weights_emb = positional_encoding(torch.cat([global_densities[:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], global_weight_emb[:, mask]], dim=-1), 6)
                fusion_feat = self.gru(gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2),
                                       global_gaussians[:, mask].unsqueeze(2),
                                       input_weights_emb,
                                       hidden_weights_emb).squeeze(2)

                global_gaussians = torch.cat([global_gaussians[:, ~mask], fusion_feat], dim=1)
                weights_0 = global_densities[:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                global_coords = torch.cat([global_coords[:, ~mask], (global_coords[:, mask]*weights_0[...,1] +
                                                coords[0][:, i, pixel_indices][:,fusion_indices_,0,0]*weights_1[...,1]) / (weights_0[...,1]+weights_1[...,1])], dim=1)
                global_densities = torch.cat([global_densities[:, ~mask], (global_densities[:, mask] +
                                                        densities[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                global_weight_emb = torch.cat([global_weight_emb[:, ~mask], (global_weight_emb[:, mask] +
                                                        weight_emb[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                
                global_extrinsics = torch.cat([global_extrinsics[:, ~mask], (global_extrinsics[:, mask]*weights_0[...,:1] +
                                                extrinsics[:, i, None]*weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1])], dim=1)
                global_depths = torch.cat([global_depths[:, ~mask], (global_depths[:, mask]*weights_0[...,0,0] +
                                                        depths[None, i, pixel_indices][:,fusion_indices_]*weights_1[...,0,0])/(weights_0[...,0,0]+weights_1[...,0,0])], dim=1)
            
            global_gaussians = torch.cat([global_gaussians, gaussians[0][:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_coords = torch.cat([global_coords, coords[0][:,i]\
                                                [:,~fusion_mask,0,0]], dim=1)
            
            global_densities = torch.cat([global_densities, densities[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_weight_emb = torch.cat([global_weight_emb, weight_emb[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_extrinsics = torch.cat([global_extrinsics, extrinsics[:,i,None].repeat(1,(~fusion_mask).sum(),1,1)], dim=1)
            global_depths = torch.cat([global_depths, depths[None,i]\
                                                [:,~fusion_mask]], dim=1)


        return global_gaussians, global_coords, global_extrinsics, global_depths
