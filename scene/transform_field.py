import torch
from torch import nn
from einops import rearrange
import math
import sys
import os
from typing import NamedTuple
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
# from minkunet import MinkUNet
from mink.minkunet import MinkUNet
import numpy as np

class dgcnn_args(NamedTuple):
    k = 20
    emb_dims = 1024
    dropout = 0.5


def move2center(pcd):
    min_coords, _  = torch.min(pcd, 0)
    max_coords, _  = torch.max(pcd, 0)
    range_coords = max_coords - min_coords
    normalized_pcd = ((pcd - min_coords) / range_coords) * 2 - 1
    center = torch.mean(normalized_pcd, 0)

    return normalized_pcd - center



class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        
        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    
    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        
        return torch.cat(out, -1)

class TransformField(nn.Module):
    def __init__(self, W1=64, W2=256, out_dim=12, in_channels_xyz=63, in_channels_t=13, voxelsize=0.005, args=None):
        super(TransformField, self).__init__()
        self.W1 = W1
        self.W2 = W2
        self.out_dim = out_dim
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_t = in_channels_t
        self.voxel_size = voxelsize
        self.front_net_in_dim = args.position_emb_level * 2 * 3 + 3
        self.front_net = nn.Sequential(
            nn.Linear(self.front_net_in_dim, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1)
        )
        self.back_net = nn.Sequential(
            nn.Linear(self.W1+64, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1), nn.ReLU(True),
            nn.Linear(self.W1, self.W1)
        )

        self.unet = MinkUNet(self.front_net_in_dim, vsize=self.voxel_size, cr=1, cs=[32,64,128,128,64,32])

        self.network = nn.Sequential(
            nn.Linear(in_channels_t+in_channels_xyz+64, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2 + in_channels_t+in_channels_xyz+64, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.W2), nn.ReLU(True),
            nn.Linear(self.W2, self.out_dim)
        )

    def forward(self, x):
        _xyz = x[:,-3:]
        #center_xyz = move2center(_xyz)
        front_feat = self.front_net(x[:,:self.front_net_in_dim])
        pc = torch.round(_xyz / self.voxel_size).type(torch.int32).cpu().numpy()
        _, inds, inverse_map = sparse_quantize(pc, return_index=True, return_inverse=True)
        
        coord_,feat_ = (torch.cat([_xyz[inds], torch.zeros(_xyz[inds].shape[0],1).to('cuda')], 1),
                        x[:,:self.front_net_in_dim][inds])
        inputpcd = SparseTensor(feats=feat_,coords=coord_)
        out = self.unet(inputpcd)             # N*32
        _pcd = torch.zeros_like(_xyz).to('cuda')
        _pcd = out[inverse_map]
        back_feat = self.back_net(torch.cat([front_feat, _pcd],-1))
        _x = torch.cat([x[:,:-3], back_feat], -1)
        _xx = _x
        for idx, n in enumerate(self.network):
            if idx != 4:
                h = n(_x)
            else:
                h = n(torch.cat([_x, _xx], -1))
            _x = h
        out = _x
     
        return out, inputpcd.C.shape[0]




class DCTLayer(torch.nn.Module):
    def __init__(
        self,
        K: int = 19,
        T: int = 20
    ):
        """
        DCTLayer, input tensor `phi` of shape [B, 3, K] and
        input tensor `t` of shape [B, 1], compute the DCT
        trajectory according to `phi`, and output the `point`
        at time `t` of shape [B, 3]
        """
        super(DCTLayer, self).__init__()
        self.K = torch.tensor(K)
        self.T = torch.tensor(T)
        self.ktensor = torch.arange(1, K+1).unsqueeze(0) # [1, K]
        self.ktensor = self.ktensor.repeat(3,1) # [3, K]
        self.ktensor = self.ktensor.unsqueeze(0) # [1, 3, K]
        self.register_buffer("ki", self.ktensor) # [1, 3, K]
        self.register_buffer("numK", self.K)
        self.register_buffer("numT", self.T)
        self.register_buffer("pi",  torch.tensor(math.pi))



