""" Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L1395-L1421 """

import torch
import torch.nn as nn
from .decoder import CHARGEDIFFNet
from utils.util import NoiseLevelEncoding

# Denoising network for ChargeDIFF
class DiffusionNet(nn.Module):
    
    def __init__(self, df_conf=None, target_property=None):
        """ init method """
        super().__init__()

        self.target_property = target_property
        self.time_emb_dim = df_conf.model.params['time_emb_dim']
        decoder_params = df_conf.decoder.params
        
        self.diffusion_net = CHARGEDIFFNet(**decoder_params)
        
        # Conditioning on categorical property (multi-hot)
        if target_property == 'chemical_system':
            self.prop_embedding = torch.nn.Linear(in_features = 17, out_features=self.time_emb_dim)
            
        # Conditioning on categorical property (one-hot)
        if target_property == 'space_group':
            self.prop_embedding = torch.nn.Linear(in_features = 230, out_features=self.time_emb_dim)
            
        # Conditioning on numeric property
        elif target_property in ['bandgap', 'energy_above_hull', 'magnetic_density', 'density']:
            self.prop_embedding = NoiseLevelEncoding(self.time_emb_dim)

        
    def forward(self, time_emb, atom_types, frac_coords, lattices, charge_dens, num_atoms, node2graph, c_concat: list = None):
        
        # for unconditoinal generation    
        if self.target_property is None:
            pred_a, pred_x, pred_l, pred_c= self.diffusion_net(time_emb, atom_types, frac_coords, lattices, charge_dens, num_atoms, node2graph)
        
        # for conditoinal generation    
        else:
            # Concatenate property embedding to timestep embedding    
            prop_emb = self.prop_embedding(c_concat[0])
            z_emb = torch.cat([time_emb, prop_emb], dim=-1)                          
            pred_a, pred_x, pred_l, pred_c = self.diffusion_net(z_emb, atom_types, frac_coords, lattices,  charge_dens, num_atoms, node2graph)
            
        return pred_a, pred_x, pred_l, pred_c