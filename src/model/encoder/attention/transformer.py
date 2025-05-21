import torch
import torch.nn as nn


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