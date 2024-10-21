import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F

from modules.layers import BasicBlock
from sr_utils.generic_utils import upsample
from einops import *


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers


class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                scales=range(4), 
                num_output_channels=1,  
                use_skips=True,
                near=0.5,
                far=15.0,
                num_samples=64,
                n_levels=-1,
                use_planes=True,
                log_plane=False,
                wo_msd=False,
                num_context_views=2,
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_samples = num_samples
        self.n_levels = n_levels

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])
        self.use_planes = use_planes
        self.log_plane = log_plane
        self.near = near
        self.far = far
        self.convs = nn.ModuleDict()
        max_depth = 2 + 2 * (not wo_msd)
        self.max_depth = max_depth
        for j in range(1, max_depth+1):
            max_i = max_depth - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != max_depth:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                self.convs[f"output_{i}"] = nn.Sequential(
                        BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                        nn.Conv2d(num_ch_out, self.num_output_channels if use_planes else 1*(i>0)+(i==0)*self.num_output_channels, 1),
                        )

        if use_planes:
            if log_plane:
                depth_candi_curr = (
                    torch.log(torch.tensor(near))
                    + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0)
                    * torch.log(torch.tensor(far / near))
                )
            else:
                min_depth = 1.0 / far
                max_depth = 1.0 / near
                depth_candi_curr = (
                    max_depth
                    + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0)
                    * (min_depth - max_depth)
                )
            self.depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]

            self.conv_depth = nn.ModuleDict()
        
            for i in range(4):
                self.conv_depth[f'{i}'] = nn.Sequential(
                                    BasicBlock(self.num_output_channels, num_samples),
                                    nn.Conv2d(num_samples, num_samples, 1),
                                    )

        self.conv_last = nn.Sequential(
                        BasicBlock(self.num_output_channels, 128),
                        nn.Conv2d(128, self.num_output_channels, 1),
                        )
        
    
    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, self.max_depth+1):
            max_i = self.max_depth - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != self.max_depth:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]
                
                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]
                depth_outputs[f"output_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]

        for i in range(self.max_depth-1,-1,-1):
            depth_planes = F.softmax(self.conv_depth[f'{i}'](depth_outputs[f"output_pred_s{i}_b1hw"]), dim=1)
            coarse_disps = (self.depth_candi_curr.to(depth_planes.device) * depth_planes).sum(dim=1, keepdim=True)
            depth_outputs[f'depth_pred_s{i}_b1hw'] = torch.exp(coarse_disps) if self.log_plane else 1.0 / coarse_disps
            depth_outputs[f'log_depth_pred_s{i}_b1hw'] = coarse_disps if self.log_plane else torch.log(1.0 / coarse_disps+1e-8)

        if self.use_planes:
            fine_disps = F.interpolate(
                coarse_disps,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            depth_map = torch.exp(fine_disps)
            depth_outputs['depth_pred_s-1_b1hw'] = depth_map
        else:
            fine_log_depth = F.interpolate(
                log_depth,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            depth_map = torch.exp(fine_log_depth)
            depth_outputs['depth_pred_s-1_b1hw'] = depth_map
        normed_depth = depth_map / depth_map.max()
        weights = torch.exp(-1.0 * (normed_depth**2) / 0.72).detach()
        depth_outputs['depth_weights'] = weights
        depth_outputs[f"output_pred_s-1_b1hw"] = self.conv_last(upsample(depth_outputs[f"output_pred_s0_b1hw"]))
        depth_outputs['depth_weights'] = F.interpolate(depth_planes,
                                                        scale_factor=2,
                                                        mode="bilinear",
                                                        align_corners=True,
                                                    ).max(dim=1, keepdim=True)[0]
        
        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1 if i == 0 else 2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)
        return outputs


class GRU(nn.Module):
    def __init__(self, input_channel=64, hidden_channel=64, weights_dim=24):
        super(GRU, self).__init__()
        self.mlp_z = nn.Sequential(nn.Linear(hidden_channel + input_channel + 2*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))
        self.mlp_r = nn.Sequential(nn.Linear(hidden_channel + input_channel + 2*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))
        self.mlp_n = nn.Sequential(nn.Linear(hidden_channel + input_channel + 1*weights_dim, hidden_channel),
                                   nn.ReLU(),
                                   nn.Linear(hidden_channel, hidden_channel))

    def forward(self, input_feat, hidden_feat, input_weights_emb, hidden_weights_emb):
        if len(input_feat.size()) == 2 and input_feat.size(0) == 1:
            input_feat = input_feat.unsqueeze(1)
        if hidden_feat is None:
            hidden_feat = torch.zeros_like(input_feat)
        input_feat_1 = torch.cat((input_feat, input_weights_emb), dim=-1)
        hidden_feat_1 = torch.cat((hidden_feat, hidden_weights_emb), dim=-1)
        concat_input = torch.cat((hidden_feat_1, input_feat_1), dim=-1)  # B x N x C
        r = torch.sigmoid(self.mlp_r(concat_input))
        z = torch.sigmoid(self.mlp_z(concat_input))
        update_feat = torch.cat((r * hidden_feat, input_feat_1), dim=-1)
        q = torch.tanh(self.mlp_n(update_feat))
        output = (1 - z) * hidden_feat + z * q
        return output



class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation = False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        try:
            return self.net(x)
        except:
            print('x.shape:', x.shape)
            print('x:', x)
            print('self.net:', self.net)
            raise ValueError

