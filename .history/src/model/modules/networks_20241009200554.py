import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork

from modules.layers import BasicBlock
from sr_utils.generic_utils import upsample


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
                use_skips=True
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
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

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                # print('+++++++++++++++++num_output_channels:', num_output_channels)
                self.convs[f"output_{i}"] = nn.Sequential(
                        BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                        nn.Conv2d(num_ch_out, self.num_output_channels, 1),
                        )

    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]
                
                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))

                outputs += [output]

                depth_outputs[f"log_output_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]

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
        return self.net(x)