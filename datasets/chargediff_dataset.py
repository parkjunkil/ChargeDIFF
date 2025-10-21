import numpy as np
from termcolor import colored, cprint

import os

import torch
import torchvision.transforms as transforms
from datasets.base_dataset import BaseDataset
from torch_geometric.data import Data

from collections import OrderedDict
from utils.data_util import (
    preprocess, add_scaled_lattice_prop, get_scaler_from_data_list, atom_classification)
import utils
import warnings

max_atom_num = 100

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class CHARGEDIFF_Dataset(BaseDataset):

    def initialize(self, opt, phase='train'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        # Assign paths for Charge Density & Structure data files
        self.dataroot = opt.dataroot
        self.dataset_mode = opt.dataset_mode

        # Assign target property for conditional models
        if opt.model == 'chargediff_uncond':
            self.target = None
        elif opt.model == 'chargediff_chemical_system':
            self.target = 'chemical_system'           
        elif opt.model == 'chargediff_bandgap':    
            self.target = 'band_gap'
        elif opt.model == 'chargediff_magnetic_density':    
            self.target = 'magnetic_density'        
        elif opt.model == 'chargediff_density':    
            self.target = 'density'        
        elif opt.model == 'vqvae':
            self.target = None    
        else:
            raise ValueError("Unexpected model type.")

        self.properties = ['band_gap', 'magnetic_density', 'density', 'chemsys']

        # Parameters for structure data processing
        self.niggli = opt.niggli
        self.primitive = opt.primitive
        self.graph_method = opt.graph_method
        self.preprocess_workers = opt.preprocess_workers
        self.tolerance = opt.tolerance
        self.use_space_group = opt.use_space_group
        self.lattice_scale_method = opt.lattice_scale_method
        
        # For Charge Density data processing
        CHGDEN_dir = f'{self.dataroot}/{self.dataset_mode}/charge_density'       
        file_list = f'{self.dataroot}/{self.dataset_mode}/splits/{phase}_split_{self.dataset_mode}.txt'

        # Preprocess Structure Data
        struc_path =  f'{self.dataroot}/{self.dataset_mode}/structure/{phase}.csv'
        struc_save_path =  f'{self.dataroot}/{self.dataset_mode}/structure/{phase}_ori.pt'
        
        self.struc_path = struc_path
        self.struc_save_path = struc_save_path
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.preprocess()
        
        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)
  
        self.lattice_scaler = None
                                                                
        if self.target in [ 'band_gap', 'magnetic_density', 'density']:
            self.scaler = get_scaler_from_data_list(self.cached_data, key=self.target)
                           
        # Load appropriate property data for each model 
        self.data_list = []
        
        for item in self.cached_data:
                            
            data_list = []
            
            mp_id = item['mp_id']
            chgden_path = f'{CHGDEN_dir}/{mp_id}.npy'
            
            if self.target == None:
                data_list.append({'mp_id':mp_id, 'chgden':chgden_path, 'struc':item})
            elif self.target == 'band_gap':
                target_prop = item['band_gap']
                data_list.append({'mp_id':mp_id, 'chgden':chgden_path, 'struc':item, 'band_gap':target_prop})   
            elif self.target == 'magnetic_density':
                target_prop = item['magnetic_density']
                data_list.append({'mp_id':mp_id, 'chgden':chgden_path, 'struc':item, 'magnetic_density':target_prop})   
            elif self.target == 'density':
                target_prop = item['density']
                data_list.append({'mp_id':mp_id, 'chgden':chgden_path, 'struc':item, 'density':target_prop})
            elif self.target == 'chemical_system':
                target_prop = item['chemsys']
                data_list.append({'mp_id':mp_id, 'chgden':chgden_path, 'struc':item, 'chemsys':target_prop})    

            self.data_list += data_list

        # Shuffle data
        np.random.default_rng(seed=0).shuffle(self.data_list)

        self.data_list = self.data_list[:self.max_dataset_size]

        cprint(f'[*] %d samples loaded for {phase}.' % (len(self.data_list),), 'yellow')
        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        
        
    def preprocess(self): 
        # Check if the processed data is already available (.pt files in ./data/MP-20-Charge/structure/)
        if os.path.exists(self.struc_save_path):
            self.cached_data = torch.load(self.struc_save_path)
        # If not, process data for training (this requires only once, and would save .pt file in ./data/MP-20-Charge/structure/)
        else:
            cached_data = preprocess(
            self.struc_path,
            num_workers=self.preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            property_list=self.properties,
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, self.struc_save_path)
            self.cached_data = cached_data


    def __getitem__(self, index):

        mp_id = self.data_list[index]['mp_id']
        
        # Process Charge Density Data
        ## Retrieve and reshape
        chgden_file = self.data_list[index]['chgden']
        chgden = np.load(chgden_file).astype(np.float32)
        chgden = torch.Tensor(chgden).unsqueeze(0)

        ## Clip and Normalize Charge Density
        thres_max_tot = self.opt.trunc_thres_max_tot
        thres_min_tot = self.opt.trunc_thres_min_tot

        chgden = torch.clamp(chgden, min=thres_min_tot, max=thres_max_tot)
        chgden = (np.log10(chgden) - np.log10(thres_min_tot)) / (np.log10(thres_max_tot) - np.log10(thres_min_tot))

        #------------------------------------------------
        # Retrieve Structure Data
        struc_data = self.data_list[index]['struc']
        
        ## Property
        if self.target in ['band_gap', 'magnetic_density', 'density']:
            prop = self.scaler.transform(self.data_list[index][self.target])
        
        elif self.target == 'chemical_system':
            prop = self.data_list[index]['chemsys']
            prop = atom_classification(prop)
           
        else:
            prop = None               
                            
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = struc_data['graph_arrays']

        struc = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        ret = {
            'id' : mp_id,
            'chgden': chgden,
            'struc' : struc,
            'prop' : prop
        }
                
        return ret

    def __len__(self):
        return self.N

    def name(self):
        return f'CHARGEDIFF_Dataset-{self.dataset_mode}'
