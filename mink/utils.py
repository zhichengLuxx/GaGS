"""
General utility functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,re,sys,json,yaml,random, argparse, torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation

from sklearn.neighbors import NearestNeighbors
import torchsparse.nn.functional as F
from torchsparse import SparseTensor, PointTensor
from torchsparse.nn.utils import get_kernel_offsets
from typing import Union, Tuple


_EPS = 1e-7  # To prevent division by zero

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point',
           'range_to_point','point_to_range']

class Logger:
    def __init__(self, path):
        self.path = path
        self.fw = open(self.path+'/log','a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()

def save_obj(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_config(path):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    
    config = dict()
    for key, value in cfg.items():
        for k,v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]



def initial_voxelize(z: PointTensor, after_res) -> SparseTensor:

    # In fact, we have already done a voxel transfer 
    # during the construction of the dataset between voxel transfers
    # This is to ensure that the voxel size is the same as the model size
    new_float_coord = torch.cat(
        [z.C[:, :3]  / after_res, z.C[:, -1].view(-1, 1)], 1)

    # hash module implemented in c++ using pybind11
    # todo: changed floor to round here
    pc_hash = F.sphash(torch.round(new_float_coord).int())

    # turned into a unique hash to obtain a unique voxel coordinate
    sparse_hash = torch.unique(pc_hash)

    # Obtain the index of pc hash in sparse hash
    idx_query = F.sphashquery(pc_hash, sparse_hash)

    counts = F.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = F.spvoxelize(torch.round(new_float_coord), idx_query,counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)

    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor.to(z.F.device)


def point_to_voxel(x: SparseTensor, z: PointTensor) -> SparseTensor:
    if z.additional_features is None or z.additional_features['idx_query'] is None \
            or z.additional_features['idx_query'].get(x.s) is None:

        pc_hash = F.sphash(
            torch.cat([
                torch.round(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)

        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts

    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps 

    return new_tensor


def voxel_to_point(x: SparseTensor,z: PointTensor, nearest=False) -> torch.Tensor:

    if z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None \
            or z.weights.get(x.s) is None:

        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)

        # kernel hash, which generates an encoding of the current coordinate including eight offsets
        old_hash = F.sphash(
            torch.cat([
                torch.round(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)


        pc_hash = F.sphash(x.C.to(z.F.device))

        idx_query = F.sphashquery(old_hash, pc_hash)

        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()

        idx_query = idx_query.transpose(0, 1).contiguous()

        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1

        new_feat = F.spdevoxelize(x.F, idx_query, weights)

        # if x.s == (1,1,1):
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))

    return new_feat

def range_to_point(x,px,py):

    r2p = []

    # todo: If we want to speed up here, we can only do a fixed number of training points 
    # done: speed increase a little in testing, but not very obviously
    # t1 = time.time() #0.01*batch_size
    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        pypx = torch.stack([p_x,p_y],dim=2).to(px[0].device)
        # print(pypx.shape,x.shape) # torch.Size([1, 111338, 2]) torch.Size([1, 32, 64, 2048])
        resampled = grid_sample(x[batch].unsqueeze(0).float(),pypx.unsqueeze(0).float())
        # print(resampled.shape) # torch.Size([1, 32, 1, 111338])
        r2p.append(resampled.squeeze().permute(1,0))
        # print(resampled.squeeze().permute(1,0).shape)

    # print(time.time()-t1) # 0.2s batch=12

    return torch.concat(r2p,dim=0)


def point_to_range(range_shape,pF,px,py):
    H, W = range_shape
    cnt = 0
    r = []
    # t1 = time.time()
    for batch,(p_x,p_y) in enumerate(zip(px,py)):
        image = torch.zeros(size=(H,W,pF.shape[1])).to(px[0].device) 
        image_cumsum = torch.zeros(size=(H,W,pF.shape[1])).to(px[0].device)+1e-5

        p_x = torch.floor((p_x/2. + 0.5) * W).long()
        p_y = torch.floor((p_y/2. + 0.5) * H).long()

        ''' v1: directly assign '''
        # image[p_y,p_x] = pF[cnt:cnt+p_x.shape[1]]

        ''' v2: use average '''
        image[p_y,p_x] += pF[cnt:cnt+p_x.shape[1]]
        image_cumsum[p_y,p_x] += torch.ones(pF.shape[1]).to(px[0].device)
        image = image/image_cumsum.to(px[0].device)
        r.append(image.permute(2,0,1))
        cnt += p_x.shape[1]
        # batch_id = torch.zeros(p_x.shape[1]).to(px[0].device)
        # p_x = torch.floor((p_x/2. + 0.5) * (W-1)).long().squeeze(0)
        # p_y = torch.floor((p_y/2. + 0.5) * (H-1)).long().squeeze(0)
        # # print(p_x.shape)
        # pxpy = torch.stack([batch_id,p_x,p_y],dim=1).to(px[0].device).int()
        # print(pxpy)
        # cm = rnf.map_count(pxpy, 1, H, W)
        # print(cm)
        # fm = rnf.denselize(pF[cnt:cnt+p_x.shape[1]], cm, pxpy)
        # print(fm)
        # # print('fm',fm.shape)
        # r.append(fm)
        # cnt += p_x.shape[1]
    # print(time.time()-t1) # 0.03s batch=12
    # print(torch.stack(r,dim=0).shape)
    return torch.stack(r,dim=0).to(px[0].device)
    # result = torch.cat(r,dim=0).to(px[0].device)
    # print(result)
    # return result