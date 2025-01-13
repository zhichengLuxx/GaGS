import time
import sys
import os

from .voxelBlock_fcgf import DownVoxelStage,UpVoxelStage,BasicDeconvolutionBlock,ResidualBlock,UpVoxelStage_withoutres
from .utils import initial_voxelize, voxel_to_point
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsparse import PointTensor, SparseTensor
import torchsparse.nn as spnn

class MinkUNet(nn.Module):
    def __init__(self,num_feats,vsize=0.05,**kwargs):
        super(MinkUNet, self).__init__()

        self.input_channel = num_feats
        self.vsize = vsize
        self.cr = kwargs.get('cr')
        self.cs = torch.Tensor(kwargs.get('cs'))
        self.cs = (self.cs*self.cr).int()
        self.output_channel = 64

        ''' voxel branch '''
        # self.voxel_stem = nn.Sequential(
        #     spnn.Conv3d(1, self.cs[0], kernel_size=5, stride=1),
        #     spnn.BatchNorm(self.cs[0]), spnn.ReLU(True),
        #     spnn.Conv3d(self.cs[0], self.cs[0], kernel_size=3, stride=1),
        #     spnn.BatchNorm(self.cs[0]), spnn.ReLU(True))
        self.voxel_init = DownVoxelStage(self.input_channel,self.cs[0],
                                      b_kernel_size=5,b_stride=1,b_dilation=1,
                                      kernel_size=3,stride=1,dilation=1)
        self.voxel_down1 = DownVoxelStage(self.cs[0], self.cs[1],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        self.voxel_down2 = DownVoxelStage(self.cs[1], self.cs[2],
                                      b_kernel_size=3, b_stride=2, b_dilation=1,
                                      kernel_size=3, stride=1, dilation=1)
        # self.voxel_down4 = DownVoxelStage(self.cs[2], self.cs[3],
        #                               b_kernel_size=3, b_stride=2, b_dilation=1,
        #                               kernel_size=3, stride=1, dilation=1)
        self.voxel_bottle = nn.Sequential(
            # BasicDeconvolutionBlock(self.cs[3], self.cs[4],
            #                         kernel_size=3, stride=2),
            ResidualBlock(self.cs[2], self.cs[3],
                          kernel_size=3, stride=1)
        )
        self.voxel_up1 = UpVoxelStage(self.cs[3],self.cs[4],self.cs[2],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        self.voxel_up2 = UpVoxelStage(self.cs[4],self.cs[5],self.cs[1],
                                 b_kernel_size=3,b_stride=2,
                                 kernel_size=3,stride=1,dilation=1)
        # self.voxel_up3 = UpVoxelStage_withoutres(self.cs[6],self.cs[7],self.cs[0],
        #                                          kernel_size=3,stride=1)
        self.voxel_final = spnn.Conv3d(self.cs[5],self.output_channel,
                                       kernel_size=1,stride=1,bias=True)

        

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def forward(self,lidar):
        points = PointTensor(lidar.F,lidar.C.float())
        # print(points.F.shape)
        v0 = initial_voxelize(points,self.vsize)
        # print(v0.F)
        # print(v0.C)

        voxel_s1 = self.voxel_init(v0)
        voxel_s2 = self.voxel_down1(voxel_s1)
        voxel_s4 = self.voxel_down2(voxel_s2)
        
        voxel_s4_tr = self.voxel_bottle(voxel_s4)
        voxel_s2_tr = self.voxel_up1(voxel_s4_tr, voxel_s4)
        voxel_s1_tr = self.voxel_up2(voxel_s2_tr, voxel_s2)
        voxel_out_final = self.voxel_final(voxel_s1_tr)

        out = voxel_to_point(voxel_out_final, points)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True))

        return out





