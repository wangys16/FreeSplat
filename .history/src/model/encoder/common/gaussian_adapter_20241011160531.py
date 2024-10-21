from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance

import numpy as np


import cv2
def norm_np(v):
    normalized_v = v / (torch.sqrt(torch.sum(v**2)+1e-8)+1e-8)
    return normalized_v
class Create_from_depth_map():
    # This class is implemented via np package, so this can only handle batch_size=1.
    # If want to handle multi-batch_size, we need to implement this via torch package.
    def __init__(self, intrinsic_matrix, height=960, width=1280, depth_trunc=12.0):
        '''

        :param intrinsic_matrix: np.array size of (3,3)
        :param height:
        :param width:
        '''
        self.intrinsic_matrix = intrinsic_matrix
        self.focal_length = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1])
        self.principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.height = height
        self.width = width
        self.depth_trunc = depth_trunc
        self.cam_pos_cam_coord = torch.tensor([0, 0, 0])
        i_coords, j_coords = torch.meshgrid(torch.range(0,height-1), torch.range(0,width-1), indexing='ij')
        coord_mat = torch.cat((j_coords[..., None], i_coords[..., None]), dim=-1)
        principal_point_mat = torch.tensor([self.principal_point[0], self.principal_point[1]])
        principal_point_mat = principal_point_mat.reshape(1, 1, 2)
        focal_length_mat = torch.tensor([self.focal_length[0], self.focal_length[1]])
        focal_length_mat = focal_length_mat.reshape(1, 1, 2)
        self.focal_length_mat = focal_length_mat.to(intrinsic_matrix.device)
        self.principal_point_mat = principal_point_mat.to(intrinsic_matrix.device) 
        self.one_mat = torch.ones((height, width, 1)).to(intrinsic_matrix.device) 
        self.cam_coord_part = ((coord_mat - principal_point_mat) / focal_length_mat).to(intrinsic_matrix.device)


    def project(self, depth_map, extrinsic_c2w_matrix):
        '''
        This for one image once.
        :param depth_map: np.array (H, W)
        :param extrinsic_c2w_matrix: np.array (4,4)
        :return:
        '''
        z_mat = depth_map.reshape(self.height, self.width, 1)
        masked_depth_map = torch.zeros_like(z_mat, device=z_mat.device)
        masked_depth_map[(z_mat < self.depth_trunc) & (z_mat > 0)] = 1
        point_mask = masked_depth_map 

        # do the computation
        cam_coord_xy = self.cam_coord_part * z_mat
        cam_coord_xyz = torch.cat([cam_coord_xy, z_mat], dim=-1)  # (h, w, 3)
        cam_coord_xyz1 = torch.cat([cam_coord_xyz, self.one_mat], dim=-1)  # (h, w, 4)
        # obtain the world_coordinate
        xyz1_cam_coord = cam_coord_xyz1.permute(2, 0, 1)  # (4, h, w)
        xyz1_cam_coord = xyz1_cam_coord.reshape(4, self.height * self.width)  # (4, h*w)
        world_coord_mat1 = torch.matmul(extrinsic_c2w_matrix, xyz1_cam_coord)  # (4, h*w)
        world_coord_mat = world_coord_mat1[:3, :]  # (3, h*w)
        world_coord_mat = world_coord_mat.T  # (h*w, 3)
        points_dir_world_coord = self.obtain_point_dir(cam_coord_xyz, extrinsic_c2w_matrix)  # (h*w, 3)

        # reshape the size of point_mask from (h, w, 1) to (h*w, 1)
        point_mask = point_mask.reshape(self.height*self.width, 1)
        
        post_xy_coords = torch.matmul(extrinsic_c2w_matrix.inverse(), world_coord_mat1)[:3]
        post_xy_coords = (post_xy_coords / post_xy_coords[2:3, :])[:2].permute(1, 0)
        post_xy_coords = post_xy_coords * self.focal_length_mat.reshape(1,2) + self.principal_point_mat

        return world_coord_mat

    def obtain_point_dir(self, cam_coord_xyz, extrinsic_c2w_m):
        h, w, _ = cam_coord_xyz.shape
        points_dir = cam_coord_xyz  # (h, w, 3)
        points_dir = points_dir.reshape(h*w, 3)
        points_dir_normed = norm_np(points_dir)
        points_dir_normed = points_dir_normed.T  # (3, h*w)
        # change the dir under world coordinates
        cam2world_mat3x3 = extrinsic_c2w_m[:3, :3]
        points_dir_world_coor = torch.matmul(cam2world_mat3x3, points_dir_normed)  # (3, h*w)
        return points_dir_world_coor.T  # (h*w, 3)

    def read_depth(self, filepath):
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000
        return depth_im


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    depth_sup: bool = False
    load_depth: bool = False



class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_gaussians,
        image_shape: tuple[int, int],
        load_depth: bool = False,
        eps: float = 1e-8,
        fusion: bool = False,
        coords = None,
    ):
        device = extrinsics.device
        h, w = image_shape
        # print('depth_sup:', self.cfg.depth_sup)
        if coords is None:
            origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)


        if not fusion:
            scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

            # Map scale features to valid scale range.
            scale_min = self.cfg.gaussian_scale_min
            scale_max = self.cfg.gaussian_scale_max
            scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
            # print('depths.shape:', depths.shape)
            # print('scales.shape:', scales.shape)
            # print('multiplier.shape:', multiplier.shape)
            scales = scales * depths[..., None] * multiplier[..., None]

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

            # Apply sigmoid to get valid colors.
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

            # Create world-space covariance matrices.
            covariances = build_covariance(scales, rotations)
            c2w_rotations = extrinsics[..., :3, :3]
            covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        
        if coords is None:
            means = torch.zeros_like(directions, device=directions.device)
            b, v = intrinsics.shape[:2]
            depth_to_pcd_now = Create_from_depth_map(intrinsic, height=h, width=w,
                                            depth_trunc=15)
            for i in range(b):
                intrinsic = intrinsics[i,0].clone().view(3,3)
                intrinsic[:1,:] *= w
                intrinsic[1:2,:] *= h
                for j in range(v):
                    pcd_w1_now = depth_to_pcd_now.project(depths[i,j].view(h,w), extrinsics[i,j].view(4,4))
                    means[i,j] = pcd_w1_now[:,None,None]




                # means = origins + directions * depths[..., None]
            # print('means.shape:', means.shape)
            # exit(0)
            if fusion:
                return means
        # elif self.cfg.depth_sup:
        #     # print('depth_sup!!!!!!!!!!!!')
        #     # print('depths.shape:', depths.shape)
        #     new_means = torch.zeros_like(directions, device=directions.device)
        #     b, v = intrinsics.shape[:2]
        #     for i in range(b):
        #         intrinsic = intrinsics[i,0].clone().view(3,3)
        #         intrinsic[:1,:] *= 320
        #         intrinsic[1:2,:] *= 240
        #         depth_to_pcd_now = Create_from_depth_map(intrinsic, height=240, width=320,
        #                                         depth_trunc=15)
        #         for j in range(v):
        #             pcd_w1_now, pcd_dir1, point_mask1 = depth_to_pcd_now.project(depths[i,j].view(240,320),\
        #                                                                         extrinsics[i,j].view(4,4))
        #             new_means[i,j] = pcd_w1_now[:,None,None]
                
        #     means = new_means
        
                

        # print('opcaities.shape:', opacities.shape)
        # print('depths.shape:', depths.shape)

        

        # b, v, r, z, y = depths.shape
        # extrinsics = extrinsics.clone().view(b, v, 4,4)
        # intrinsics = intrinsics.clone().view(b, v, 3,3)
        # intrinsics[:,:,0] *= 640
        # intrinsics[:,:,1] *= 480
        # depths = depths.view(b, v, 480, 640)
        # means = torch.zeros_like(means, device=means.device)
        # # print('means.shape:', means.shape)


        # for i in range(b):
        #     for j in range(v):
        #         intrinsic_now = intrinsics[i,j].clone()
        #         depth_to_pcd_now = Create_from_depth_map(intrinsic_now, height=480, width=640,
        #                                         depth_trunc=15)
        #         pcd_now, pcd_dir, mask = depth_to_pcd_now.project(depths[i,j], extrinsics[i,j])
        #         means[i,j] = pcd_now[:,None,None]
            
        # # print('extrinsics.shape:', extrinsics.shape)
        # # print('intrinsics.shape:', intrinsics.shape)
        # # new_pc = depth2pc(depths, extrinsics, intrinsics, image_shape)
        # # print('coordinates.shape:', coordinates.shape)
            
        # print('final intrinsic:', intrinsic)
        # torch.save({'means': means, 'depths': depths, 'origins': origins, \
        #             'extrinsics': extrinsics, 'intrinsics': intrinsics, 'image_shape': image_shape,}, 
        #             'means.th')
        # exit(0)
            
        # print('means.shape:', means.shape)
        if coords is None:
            harmonics = rotate_sh(sh, c2w_rotations[..., None, :, :])
        else:
            harmonics = sh
        # if fusion:
        #     covariances = covariances[:,-1:]
        #     harmonics = harmonics[:,-1:]
        #     opacities = opacities[:,-1:]
        #     scales = scales[:,-1:]
        #     rotations = rotations[:,-1:]
        #     means = means[:,-1:]
        # print('means.shape:', means.shape)
        # print('covariances.shape:', covariances.shape)
        # print('covariances.shape:', covariances.shape)
        # print('harmonics.shape:', harmonics.shape)
        # print('opacities.shape:', opacities.shape)
        # print('scales.shape:', scales.shape)
        # print('rotations.shape:', rotations.shape)
        # exit(0)
        return Gaussians(
            means=means if (coords is None) else coords,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
            # Note: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh
