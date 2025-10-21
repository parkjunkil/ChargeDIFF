# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import torch
import torchvision.utils as vutils

from datasets.base_dataset import CreateDataset
from datasets.dataloader import CreateDataLoader, get_data_generator

from models.base_model import create_model

from utils.util import seed_everything

#from utils.constants import CompScalerMeans, CompScalerStds

import joblib
from collections import OrderedDict

from scipy.stats import wasserstein_distance

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist

#import crystal_toolkit.components as ctc
#import dash
#from dash import html
#from dash.dependencies import Input, Output

'''
def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )
'''

############ all Opt classes ############

class BaseOpt(object):
    def __init__(self, gpu_ids=0, seed=None):
        # important args
        self.isTrain = False
        self.gpu_ids = [gpu_ids]
        # self.device = f'cuda:{gpu_ids}'
        self.device = 'cuda'
        self.debug = '0'

        # default args
        self.serial_batches = False
        self.nThreads = 4
        self.distributed = False

        # hyperparams
        self.batch_size = 1

        # dataset args
        self.max_dataset_size = 10000000
        self.trunc_thres_max = 100.0
        self.trunc_thres_min = 0.0001

        if seed is not None:
            seed_everything(seed)
            
        self.phase = 'test'

        self.vq_model = 'vqvae'
        self.vq_cfg = './configs/vqvae.yaml'
        self.vq_ckpt ='./saved_ckpt/vqvae.pth'
 
 
    def name(self):

        return 'BaseOpt'


class VQVAEOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids)

        # some other custom args here

        print(f'[*] {self.name()} initialized.')
        
        
    def init_dset_args(self, dataroot='data', dataset_mode='MP-40', res=32):
        self.dataroot = dataroot
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode

    def init_model_args(
            self,
            ckpt_path="./saved_ckpt/df.pth",
        ):
        self.model = 'vqvae_chgdiff'
        self.vq_cfg = './configs/vqvae.yaml'
        self.ckpt = ckpt_path
        
        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.preprocess_workers = 30
        self.property = 'energy_above_hull'
        self.tolerance = 0.1
        self.use_space_group = False
        self.lattice_scale_method = 'scale_length'

    def name(self):
        return 'VQVAE_TestOpt'


class ChargeDIFFOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='MP-20', res=32):
        self.dataroot = dataroot
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode

    def init_model_args(
            self,
            ckpt_path="./saved_ckpt/df.pth",
            vq_ckpt_path="./saved_ckpt/vqvae.pth",
        ):
        self.model = 'chgdiff_uncond'
        self.df_cfg = './configs/chgdiff-uncond.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae_chgdiff'
        self.vq_cfg = './configs/vqvae_11.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'MP-20'
        
        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.preprocess_workers = 30
        self.property = 'energy_above_hull'
        self.tolerance = 0.1
        self.use_space_group = False
        self.lattice_scale_method = 'scale_length'


    def name(self):
        return 'CHGDIFF_TestOption'


############ For evaluation of generated structures ############
'''
Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}
'''

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']



class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


############ END: all Opt classes ############
# get partial shape from range
def get_partial_shape(shape, xyz_dict, z=None):
    """
        args:  
            shape: input sdf. (B, 3, H, W, D)
            xyz_dict: user-specified range.
                x: left to right
                y: bottom to top
                z: front to back
    """
    x = shape
    device = x.device
    (x_min, x_max) = xyz_dict['x']
    (y_min, y_max) = xyz_dict['y']
    (z_min, z_max) = xyz_dict['z']
    
    # clamp to [-1, 1]
    x_min, x_max = max(-1, x_min), min(1, x_max)
    y_min, y_max = max(-1, y_min), min(1, y_max)
    z_min, z_max = max(-1, z_min), min(1, z_max)

    B, _, H, W, D = x.shape # assume D = H = W

    x_st = int( (x_min - (-1))/2 * H )
    x_ed = int( (x_max - (-1))/2 * H )
    
    y_st = int( (y_min - (-1))/2 * W )
    y_ed = int( (y_max - (-1))/2 * W )
    
    z_st = int( (z_min - (-1))/2 * D )
    z_ed = int( (z_max - (-1))/2 * D )
    
    # print('x: ', xyz_dict['x'], x_st, x_ed)
    # print('y: ', xyz_dict['y'], y_st, y_ed)
    # print('z: ', xyz_dict['z'], z_st, z_ed)

    # where to keep    
    x_mask = torch.ones(B, 1, H, W, D).bool().to(device)
    x_mask[:, :, :x_st, :, :] = False
    x_mask[:, :, x_ed:, :, :] = False
    
    x_mask[:, :, :, :y_st, :] = False
    x_mask[:, :, :, y_ed:, :] = False
    
    x_mask[:, :, :, :, :z_st] = False
    x_mask[:, :, :, :, z_ed:] = False
        
    shape_part = x.clone()
    shape_missing = x.clone()
    shape_part[~x_mask] = 1.0
    shape_missing[x_mask] = 1.0
    
    ret = {
        'shape_part': shape_part,
        'shape_missing': shape_missing,
        'shape_mask': x_mask,
    }
    
    if z is not None:
        B, _, zH, zW, zD = z.shape # assume D = H = W

        x_st = int( (x_min - (-1))/2 * zH )
        x_ed = int( (x_max - (-1))/2 * zH )
        
        y_st = int( (y_min - (-1))/2 * zW )
        y_ed = int( (y_max - (-1))/2 * zW )
        
        z_st = int( (z_min - (-1))/2 * zD )
        z_ed = int( (z_max - (-1))/2 * zD )
        
        # where to keep    
        z_mask = torch.ones(B, 3, zH, zW, zD).to(device)
        z_mask[:, :, :x_st, :, :] = 0.
        z_mask[:, :, x_ed:, :, :] = 0.
        
        z_mask[:, :, :, :y_st, :] = 0.
        z_mask[:, :, :, y_ed:, :] = 0.
    
        z_mask[:, :, :, :, :z_st] = 0.
        z_mask[:, :, :, :, z_ed:] = 0.
        
        ret['z_mask'] = z_mask

    return ret


def get_partial_shape_complex(shape, xyz_dict_list, z=None):
    """
        args:  
            shape: input charge density. (B, 1, H, W, D)
            xyz_dict: user-specified range.
                x: left to right
                y: bottom to top
                z: front to back
    """
    x = shape
    device = x.device
    
    B, _, zH, zW, zD = z.shape # assume D = H = W

    z_mask = torch.zeros(B, 3, zH, zW, zD).to(device)


    for xyz_dict in xyz_dict_list:
    
        (x_min, x_max) = xyz_dict['x']
        (y_min, y_max) = xyz_dict['y']
        (z_min, z_max) = xyz_dict['z']
        
        # clamp to [-1, 1]
        x_min, x_max = max(-1, x_min), min(1, x_max)
        y_min, y_max = max(-1, y_min), min(1, y_max)
        z_min, z_max = max(-1, z_min), min(1, z_max)


        x_st = int( (x_min - (-1))/2 * zH )
        x_ed = int( (x_max - (-1))/2 * zH )
        
        y_st = int( (y_min - (-1))/2 * zW )
        y_ed = int( (y_max - (-1))/2 * zW )
        
        z_st = int( (z_min - (-1))/2 * zD )
        z_ed = int( (z_max - (-1))/2 * zD )
        
        # where to keep    
        
        z_mask[:, :, x_st:x_ed, y_st:y_ed, z_st:z_ed] = 1.

    return z_mask