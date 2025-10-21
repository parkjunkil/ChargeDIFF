# Reference: A large portion of the model architecture is adapted from the repos: https://github.com/yccyenchicheng/SDFusion, https://github.com/jiaor17/DiffCSP

import os
from collections import OrderedDict
from omegaconf import OmegaConf
from termcolor import colored, cprint
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from models.base_model import BaseModel
from models.networks.diffusion_networks.network import DiffusionNet
from models.model_utils import load_vqvae
from utils.util import BetaScheduler, SigmaScheduler, d_log_p_wrapped_normal, D3PM, NoiseLevelEncoding
from utils.data_util import lattice_params_to_matrix_torch, get_crystal_array_list, lattices_to_params_shape
from utils.data_util import Crystal, SampleDataset
from torch_geometric.data import DataLoader
from utils.distributed import reduce_loss_dict # distribution
from utils.util_3d import render_isosurface # rendering

class CHARGEDIFF_Model(BaseModel):
    def name(self):
        return 'CHARGEDIFF-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.max_atom_num = 100
                
        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        df_conf = OmegaConf.load(opt.df_cfg)
        vqvae_conf = OmegaConf.load(opt.vq_cfg)

        # aasign latent charge density dimension for latent diffuison
        ddconfig = vqvae_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)
        
        # assign target property for conditional generation
        if 'uncond' in opt.model:
            self.target_property = None
        elif opt.model == 'chargediff_bandgap':
            self.target_property = 'bandgap'
        elif opt.model == 'chargediff_magnetic_density':
            self.target_property = 'magnetic_density'
        elif opt.model == 'chargediff_density':
            self.target_property = 'density'
        elif opt.model == 'chargediff_chemical_system':
            self.target_property = 'chemical_system'
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)
                
        # init diffusion networks
        self.df = DiffusionNet(df_conf=df_conf, target_property=self.target_property)
        self.df.to(self.device)

        # init vqvae from pre-trained checkpoint
        self.vqvae = load_vqvae(vqvae_conf, vq_ckpt=opt.vq_ckpt, opt=opt)
        
        # Set up timestep embeddings
        self.timesteps = df_conf.model.params['timesteps']
        self.time_emb_dim = df_conf.model.params['time_emb_dim'] 
        self.time_embedding = NoiseLevelEncoding(self.time_emb_dim)    
        
        ## Scheduler set up
        # atom_type (A)
        self.beta_schedule_mode_A = df_conf.model.params['atom_scheduler']['beta_schedule_mode']
        self.beta_begin_A = df_conf.model.params['atom_scheduler']['beta_begin']
        self.beta_end_A = df_conf.model.params['atom_scheduler']['beta_end']

        # coordinate (X)
        self.sigma_begin_X = df_conf.model.params['coordinate_scheduler']['sigma_begin']
        self.sigma_end_X = df_conf.model.params['coordinate_scheduler']['sigma_end']
        
        # lattice (L)
        self.beta_schedule_mode_L = df_conf.model.params['lattice_scheduler']['beta_schedule_mode']
        self.beta_begin_L = df_conf.model.params['lattice_scheduler']['beta_begin']
        self.beta_end_L = df_conf.model.params['lattice_scheduler']['beta_end']
        
        # charge (C)
        self.beta_schedule_mode_C = df_conf.model.params['charge_scheduler']['beta_schedule_mode']
        self.beta_begin_C = df_conf.model.params['charge_scheduler']['beta_begin']
        self.beta_end_C = df_conf.model.params['charge_scheduler']['beta_end']

        ## Coefficients for combined loss        
        self.cost_coord = df_conf.model.params['cost_coord']
        self.cost_lattice= df_conf.model.params['cost_lattice']
        self.cost_atom = df_conf.model.params['cost_atom']
        self.cost_chgden = df_conf.model.params['cost_chgden']
        self.d3pm_hybrid_coeff = df_conf.model.params['d3pm_hybrid_coeff'] 
               
        self.init_diffusion_params(opt=opt)

        
        ######## END: Define Networks ########
        
        # initialize optimizers and learning rate schedulers
        if self.isTrain:
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr, foreach=False)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=100, min_lr=1e-6)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]
            self.print_networks(verbose=False)

        # load from saved checkpoint if required
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
                self.schedulers = [self.scheduler]

        # for distributed training with multi-gpus
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module
            self.vqvae_module = self.vqvae.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae
            
    # for distributed training with multi-gpus
    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, opt=None):
        
        # setup schedulers for diffusion (each for A, X, L, and C)
        self.beta_scheduler_A = BetaScheduler(timesteps = self.timesteps,
                                            scheduler_mode = self.beta_schedule_mode_A,
                                            beta_start = self.beta_begin_A,
                                            beta_end = self.beta_end_A,
                                            device = self.device,)
        
        self.sigma_scheduler_X = SigmaScheduler(timesteps = self.timesteps,
                                              sigma_begin = self.sigma_begin_X,
                                              sigma_end = self.sigma_end_X,
                                              device = self.device,)
        
        self.beta_scheduler_L = BetaScheduler(timesteps = self.timesteps,
                                            scheduler_mode = self.beta_schedule_mode_L,
                                            beta_start = self.beta_begin_L,
                                            beta_end = self.beta_end_L,
                                            device = self.device,)
                
        self.beta_scheduler_C = BetaScheduler(timesteps = self.timesteps,
                                            scheduler_mode = self.beta_schedule_mode_C,
                                            beta_start = self.beta_begin_C,
                                            beta_end = self.beta_end_C,
                                            device = self.device,)

        # D3PM (discrete diffusion) for atom type
        self.d3pm = D3PM(
            beta_scheduler = self.beta_scheduler_A,
            num_timesteps = self.timesteps,
            max_atoms = self.max_atom_num,
            d3pm_hybrid_coeff = self.d3pm_hybrid_coeff,
            device = self.device,
        )
        
    ############################ END: init diffusion params ############################
    
    # load batch and assign structure information and charge density
    def set_input(self, input=None, max_sample=None):
                                
        self.batch_struc = input['struc']
        self.prop = input['prop']
        
        self.batch_chgden = input['chgden']
        
        if max_sample is not None: # not necessary
            self.batch_struc = self.batch_struc[:max_sample]
            self.batch_chgden = self.batch_chgden[:max_sample]
            self.prop = self.prop[:max_sample]
                
        vars_list = ['batch_struc', 'batch_chgden']
                
        # for conditional generation
        if self.target_property is not None: 
            vars_list.append('prop')

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()

    # run denosing netowrk
    def apply_model(self, time_emb, noisy_a, noisy_x, noisy_l, noisy_c, num_atoms, batch, cond):

        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            cond = {'c_concat': cond}

        # eps
        out = self.df(time_emb, noisy_a, noisy_x, noisy_l, noisy_c, num_atoms, batch, **cond)

        return out

    # loss function between ground-truth and predicted noise and score
    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # sample noisy state at given timestep: q(Xt|X0)
    def q_sample(self, atom_types, frac_coords, lattices, charge_dens, times, rand_a=None, rand_x=None, rand_l=None, rand_c=None):
        ## Setup Coefficients for Diffusion Process (for A, it is handled in q_sample funciton defined in d3pm)
        # X
        sigmas = self.sigma_scheduler_X.sigmas[times]
        sigmas_norm = self.sigma_scheduler_X.sigmas_norm[times]
        sigmas_per_atom = sigmas.repeat_interleave(self.batch_struc.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(self.batch_struc.num_atoms)[:, None]

        # L
        alphas_cumprod_L = self.beta_scheduler_L.alphas_cumprod[times]
        c0_l = torch.sqrt(alphas_cumprod_L)
        c1_l = torch.sqrt(1. - alphas_cumprod_L)

        # C
        alphas_cumprod_C = self.beta_scheduler_C.alphas_cumprod[times]
        c0_c = torch.sqrt(alphas_cumprod_C)
        c1_c = torch.sqrt(1. - alphas_cumprod_C)
        
        # sample random noise for each parameter
        if rand_a == None:
            rand_a = torch.rand(self.batch_struc.num_nodes, self.max_atom_num).to(self.device)
        if rand_x == None:
            rand_x = torch.randn_like(frac_coords)
        if rand_l == None:
            rand_l = torch.randn_like(lattices) 
        if rand_c == None:
            rand_c = torch.randn_like(charge_dens)

        # sample noisy state
        t_per_atom = times.repeat_interleave(self.batch_struc.num_atoms, dim=0)
        
        noisy_a = self.d3pm.q_sample(atom_types, t_per_atom, rand_a) # q(At|A0)
        noisy_x = (frac_coords + sigmas_per_atom * rand_x) % 1. # q(Xt|X0)
        rand_x_wrap = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom) # Wrapped Normal Distribution
        noisy_l = c0_l[:, None, None] * lattices + c1_l[:, None, None] * rand_l # q(Lt|L0)
        noisy_c = c0_c[:, None, None, None, None] * charge_dens + c1_c[:, None, None, None, None] * rand_c # q(Ct|C0)
        
        return noisy_a, noisy_x, noisy_l, noisy_c, rand_a, rand_x_wrap, rand_l, rand_c

    # get loss for training
    def p_losses(self, atom_types, frac_coords, lattices, charge_dens, times, cond, noise = None):

        time_emb = self.time_embedding(times)                
        t_per_atom = times.repeat_interleave(self.batch_struc.num_atoms, dim=0)

        # sample noisy state and imposed noise for each component
        noisy_a, noisy_x, noisy_l, noisy_c, rand_a, rand_x_wrap, rand_l, rand_c = self.q_sample(atom_types, frac_coords, lattices, charge_dens, times)
    
        # predict noise and score thorugh denoising network 
        pred_a, pred_x, pred_l, pred_c = self.apply_model(time_emb, noisy_a, noisy_x, noisy_l, noisy_c, self.batch_struc.num_atoms, self.batch_struc.batch, cond)
        
        ## Calculate atom loss following D3PM
        # loss for atom_types (1) VB (2) CE loss
        true_q_posterior_logits = self.d3pm.q_posterior_logits(
            atom_types, noisy_a, t_per_atom
        )
        pred_q_posterior_logits = self.d3pm.q_posterior_logits(
            pred_a, noisy_a, t_per_atom, is_x_0_one_hot=True
        )
        vb_loss = self.d3pm.categorical_kl_logits(
            true_q_posterior_logits, pred_q_posterior_logits
        )
        ce_loss = F.cross_entropy(pred_a.flatten(0, -2), atom_types.flatten())
        
        loss_atom = vb_loss + ce_loss * self.d3pm.hybrid_coeff
                
        ## Calculate loss for coordinate, lattice and charge
        loss_coord = F.mse_loss(pred_x, rand_x_wrap)
        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_chgden = F.mse_loss(pred_c, rand_c)

        ## Calculate total loss
        loss_total = self.cost_lattice * loss_lattice + self.cost_coord * loss_coord + self.cost_atom * loss_atom + self.cost_chgden * loss_chgden

        loss_dict = {}
        loss_dict['loss_tot'] = loss_total
        loss_dict['loss_atom']  = loss_atom
        loss_dict['loss_coord'] = loss_coord
        loss_dict['loss_lattice'] = loss_lattice
        loss_dict['loss_chgden']  = loss_chgden

        return pred_a, pred_x, pred_l, pred_c, loss_total, loss_dict


    def forward(self):

        self.df.train()
        batch_size = self.batch_struc.num_graphs

        # vqvae is frozen during the training
        with torch.no_grad(): 
            charge_dens = self.vqvae(self.batch_chgden, forward_no_quant=True, encode_only=True)
                
        times = self.beta_scheduler_A.uniform_sample_t(batch_size, self.device)
                  
        # read structural data and charge density data from batch
        atom_types  =  self.batch_struc.atom_types
        frac_coords =  self.batch_struc.frac_coords        
        lattices    =  lattice_params_to_matrix_torch(self.batch_struc.lengths, self.batch_struc.angles)
        c = self.prop
        
        _, _, _, _, loss, loss_dict = self.p_losses(atom_types, frac_coords, lattices, charge_dens, times, c)
        
        self.loss = loss
        self.loss_dict = loss_dict
        

    @torch.no_grad()
    def inference(self): # For evaluation

        self.df.eval()

        batch_size = self.batch_struc.num_graphs

        with torch.no_grad():
            charge_dens = self.vqvae(self.batch_chgden, forward_no_quant=True, encode_only=True)        

        times = self.beta_scheduler_A.uniform_sample_t(batch_size, self.device)
          
        atom_types  =  self.batch_struc.atom_types
        frac_coords =  self.batch_struc.frac_coords        
        lattices    =  lattice_params_to_matrix_torch(self.batch_struc.lengths, self.batch_struc.angles)
        c = self.prop
        
        _, _, _, _, loss, loss_dict = self.p_losses(atom_types, frac_coords, lattices, charge_dens, times, c)

        self.df.train()
        
        return loss, loss_dict


    @torch.no_grad()
    def eval_metrics(self, dl): # Obtain loss metrics during the evaluation

        self.switch_eval()
        
        loss_tot = []
        loss_atom = []
        loss_coord = []
        loss_lattice = []
        loss_chgden = []
        
        with torch.no_grad():
            for ix, data in tqdm(enumerate(dl), total=len(dl)):

                self.set_input(data)

                loss, loss_dict = self.inference()

                loss_tot.append(loss_dict['loss_tot'].detach().unsqueeze(0))
                loss_atom.append(loss_dict['loss_atom'].detach().unsqueeze(0))
                loss_coord.append(loss_dict['loss_coord'].detach().unsqueeze(0))
                loss_lattice.append(loss_dict['loss_lattice'].detach().unsqueeze(0))
                loss_chgden.append(loss_dict['loss_chgden'].detach().unsqueeze(0))
 
        loss_tot = torch.cat(loss_tot)
        loss_atom = torch.cat(loss_atom)
        loss_coord = torch.cat(loss_coord)
        loss_lattice = torch.cat(loss_lattice)
        loss_chgden = torch.cat(loss_chgden)

        ret = OrderedDict([
            ('loss_tot', loss_tot.mean().data),
            ('loss_atom', loss_atom.mean().data),       
            ('loss_coord', loss_coord.mean().data),
            ('loss_lattice', loss_lattice.mean().data),      
            ('loss_chgden', loss_chgden.mean().data),    
        ])

        self.switch_train()
        return ret


    @torch.no_grad()
    def sample(self, batch, c = None, x0 = None, mask = None, step_lr = 1e-5):

        # For A, L and C, ancestral predictor was applied without corrector (as in DDPM)
        # For X, Annealed Lagevin dynamics was further applied as corrector (as in SMLD)
            
        batch_size = batch.num_graphs
        num_atoms  = batch.num_atoms
        num_atoms_tot  = batch.num_nodes

        # Condition embedding
        if self.target_property != None:
            if c.device != num_atoms.device:
                c = c.to(num_atoms.device)
                
        # Sample variables at t=T 
        a_T = torch.full((num_atoms_tot,), 0).to(self.device)
        x_T = torch.rand([num_atoms_tot, 3]).to(self.device)
        l_T = torch.randn([batch_size, 3, 3]).to(self.device) 
        c_T = torch.randn([batch_size, self.z_shape[0], self.z_shape[1], self.z_shape[2], self.z_shape[3]]).to(self.device)

        traj = {self.beta_scheduler_A.timesteps: {
            'num_atoms' : num_atoms,
            'atom_types' : a_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
            'chgdens' : c_T
        }}

        # Iterative denoising process to obtain new strucutre (t=T -> t=0) 
        for t in tqdm(range(self.beta_scheduler_A.timesteps, 0, -1)):
            
            a_t = traj[t]['atom_types']
            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            c_t = traj[t]['chgdens']
            
            ## embed timestep and set required parameters
            # timestep
            times = torch.full((batch_size, ), t, device = self.device)
            t_per_atom = times.repeat_interleave(num_atoms, dim=0)
            time_emb = self.time_embedding(times)

            # X
            sigma_x = self.sigma_scheduler_X.sigmas[t]
            sigma_norm_x = self.sigma_scheduler_X.sigmas_norm[t]
           
            # L
            alphas_l = self.beta_scheduler_L.alphas[t]
            alphas_cumprod_l = self.beta_scheduler_L.alphas_cumprod[t]
            sigmas_l = self.beta_scheduler_L.sigmas[t]
            coeff_0_l = 1.0 / torch.sqrt(alphas_l)
            coeff_1_l = (1 - alphas_l) / torch.sqrt(1 - alphas_cumprod_l)

            # C
            alphas_c = self.beta_scheduler_C.alphas[t]
            beta_c = self.beta_scheduler_C.betas[t-1]
            alphas_cumprod_c = self.beta_scheduler_C.alphas_cumprod[t]
            sigmas_c = self.beta_scheduler_C.sigmas[t]
            coeff_0_c = 1.0 / torch.sqrt(alphas_c)
            coeff_1_c = (1 - alphas_c) / torch.sqrt(1 - alphas_cumprod_c)
            
            ## For Charge Density-based Inverse Design
            if mask is not None:
                assert x0 is not None
                noise = torch.randn_like(x0)
                c0_c = torch.sqrt(alphas_cumprod_c)
                c1_c = torch.sqrt(1. - alphas_cumprod_c)
                c_orig = c0_c * x0 + c1_c * noise
                c_t = c_orig * mask + c_t * (1. - mask)
                       
            ## Corrector ##
            # run decoder
            _, pred_x, _, _ = self.apply_model(time_emb, a_t, x_t, l_t, c_t, batch.num_atoms, batch.batch, c)

            # sample random noise
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            # update X
            pred_x = pred_x * torch.sqrt(sigma_norm_x)
            step_size = step_lr * (sigma_x / self.sigma_scheduler_X.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x 
            
            # set remain variables unchanged
            a_t_minus_05 = a_t
            l_t_minus_05 = l_t
            c_t_minus_05 = c_t

            ## Predictor ##
            # sample random noises
            rand_a = torch.rand((num_atoms_tot, self.max_atom_num)).to(self.device) if t>1 else torch.zeros((num_atoms_tot, self.max_atom_num)).to(self.device)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_c = torch.randn_like(c_T) if t > 1 else torch.zeros_like(c_T)

            # run decoder
            pred_a, pred_x, pred_l, pred_c = self.apply_model(time_emb, a_t_minus_05, x_t_minus_05, l_t_minus_05, c_t_minus_05, batch.num_atoms, batch.batch, c)

            # update A
            pred_x_start_logits = pred_a  # [B_n, max_atoms]
            a_t_minus_l = self.d3pm.p_logits(
                pred_x_start_logits=pred_x_start_logits,
                x_t_atom_types=a_t_minus_05,
                t_per_node=t_per_atom,
                noise=rand_a,
            )            
            
            # update X
            pred_x = pred_x * torch.sqrt(sigma_norm_x)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x 
            
            # update L
            l_t_minus_1 = coeff_0_l * (l_t_minus_05 - coeff_1_l * pred_l) + sigmas_l * rand_l 

            # update C
            c_t_minus_1 = coeff_0_c * (c_t_minus_05 - coeff_1_c * pred_c) + sigmas_c * rand_c

            traj[t - 1] = {
                'num_atoms' : num_atoms,
                'atom_types' : a_t_minus_l,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'chgdens' : c_t_minus_1                
            }

        return traj

    # save obtained structure after sampling
    @torch.no_grad()
    def diffusion(self, loader, c = None, x0 = None, mask = None, step_lr = 1e-5):
        
        frac_coords = []
        num_atoms = []
        atom_types = []
        lattices = []
        chgdens = []
        traj_list = []
        
        for _, batch in enumerate(loader):

            if torch.cuda.is_available():
                batch.cuda()

            traj = self.sample(batch = batch, step_lr = step_lr, c = c, x0 = x0, mask = mask)
            traj_list.append(traj)
            
            # index to save. index:0 -> save only the last structure. Change this if you want to save the whole trajectory
            idx_save = [0]

            for idx in idx_save:
                outputs = traj[idx]
                frac_coords.append(outputs['frac_coords'].detach().cpu())
                num_atoms.append(outputs['num_atoms'].detach().cpu())
                atom_types.append(outputs['atom_types'].detach().cpu())
                lattices.append(outputs['lattices'].detach().cpu())
                chgdens_decode = self.vqvae.decode_no_quant(outputs['chgdens']) # decode to original resolution (32,32,32) using VQ-VAE decoder
                chgdens.append(chgdens_decode.detach().cpu())


        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0) 
        atom_types[atom_types==0] = 99 # atom index 0 is an arbitruary atom during the D3PM. Mappting it to 99 to output structure with an atom index with 0
        lattices = torch.cat(lattices, dim=0)
        chgdens = torch.cat(chgdens, dim=0)
        lengths, angles = lattices_to_params_shape(lattices)

        data_out = {
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lattices': lattices,
            'lengths': lengths,
            'angles': angles,
            'chgdens': chgdens,
        }
        return data_out

    
    # Unconditional generation without any guidance
    @torch.no_grad()
    def ab_initio_gen(self, num_batch = 10, batch_size = 50, dataset='mp_20_charge', cutoff = 0.5, step_lr = 1e-5):
        
        # dataset: source to sample number of atoms for each crystal
        test_set = SampleDataset(dataset, num_batch * batch_size)
        test_loader = DataLoader(test_set, batch_size = batch_size)
                        
        data_out = self.diffusion(test_loader, step_lr = step_lr)
    
        # save generated crystals
        crys_tot = []
        crys_array_lists, _ = get_crystal_array_list(data_out, batch_idx = -2)
        
        for i in range(len(crys_array_lists)):
            crys = Crystal(crys_array_lists[i], cutoff = cutoff)
            crys_tot.append(crys)
        
        # save generated charge densities
        chgdens = data_out['chgdens']
        chgdens = [chgdens[i,0,:,:,:] for i in range(chgdens.size(0))]
        
        return crys_tot, chgdens
        
        
    # Conditional generation (e.g., bandgap, magnetic density, ...)
    @torch.no_grad()
    def cond(self, batch_size = 20, target = 10, dataset='mp_20_charge', scaler = None , cutoff = 0.5):
        
        # dataset: source to sample number of atoms for each crystal
        test_set = SampleDataset(dataset, batch_size)
        test_loader = DataLoader(test_set, batch_size = batch_size)
                          
        if scaler is not None: # load scaler for numeric properties
            target = scaler.transform(target).repeat(batch_size)
        else:
            target = target.unsqueeze(0).to(self.device)
            target = target.repeat(batch_size, 1)
                            
        data_out = self.diffusion(test_loader, c = target)
        
        # save generated crystals
        crys_tot = []
        crys_array_lists, _ = get_crystal_array_list(data_out, batch_idx = -2)
        
        for i in range(len(crys_array_lists)):
            crys = Crystal(crys_array_lists[i], cutoff = cutoff)
            crys_tot.append(crys)
            
        # save generated charge densities
        chgdens = data_out['chgdens']
        chgdens = [chgdens[i,0,:,:,:] for i in range(chgdens.size(0))]
        
        return crys_tot, chgdens
        
    
    # Conditional generation with Charge denisty-based inverse design (especially, with chemical system conditoining for transition metal oxide generation)
    # Used for the generation of cathode materials with desired ion migration pathway as described in the main text
    @torch.no_grad()
    def cond_and_charge_manipulation(self, inpaint_target, inpaint_region, batch_size = 10, dataset='mp_20', cond_target = None, scaler = None, cutoff = 0.5):
  
        from utils.demo_util import get_partial_shape
  
        # inpaint_target: target charge density for inpainting
        # for example, to obtain zero (low) charge density; shape = torch.full((1,32,32,32), 0.0)
        if inpaint_target.dim() == 4:
            inpaint_target = inpaint_target.unsqueeze(0)
            inpaint_target = inpaint_target.to(self.device)
        inpaint_target = inpaint_target.repeat(batch_size,1,1,1,1)
    
        # dataset: source to sample number of atoms for each crystal
        test_set = SampleDataset(dataset, batch_size)
        test_loader = DataLoader(test_set, batch_size = batch_size)
        
        # load vqvae to map the targeted charge density for inpainting into the latent space
        z = self.vqvae(inpaint_target, forward_no_quant=True, encode_only=True) 
        
        # inpaint_region : dictionary representing desired region for inpainting
        # convert targeted region for inpainitng into latent space coordinate
        ret = get_partial_shape(inpaint_target, xyz_dict=inpaint_region, z=z) 
        x_mask, z_mask = ret['shape_mask'], ret['z_mask']
                                  
        if scaler is not None: # load scaler for numeric properties
            target = scaler.transform(target).repeat(batch_size)
        else:
            target = target.unsqueeze(0).to(self.device)
            target = target.repeat(batch_size, 1)                              
         
        # diffusion sampling
        data_out = self.diffusion(test_loader, c = target, x0 = z, mask = z_mask)

        # save generated crystals
        crys_tot = []
        crys_array_lists, _ = get_crystal_array_list(data_out, batch_idx = -2)
        for i in range(len(crys_array_lists)):
            crys = Crystal(crys_array_lists[i], cutoff = cutoff)
            crys_tot.append(crys)
            
        # save generated charge densities
        chgdens = data_out['chgdens']            
        chgdens = [chgdens[i,0,:,:,:] for i in range(chgdens.size(0))]
        
        return crys_tot, chgdens

    # back-propagation for model training
    def backward(self):
        
        # For distributed training
        self.loss_dict = reduce_loss_dict(self.loss_dict)
        
        self.loss_total = self.loss_dict['loss_tot']
        self.loss_lattice = self.loss_dict['loss_lattice']
        self.loss_coord = self.loss_dict['loss_coord']
        self.loss_atom = self.loss_dict['loss_atom']
        self.loss_chgden = self.loss_dict['loss_chgden']

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.set_requires_grad([self.df], requires_grad=True)
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.data),
            ('atom', self.loss_atom.data),
            ('coord', self.loss_coord.data),
            ('lattice', self.loss_lattice.data),
            ('chgden', self.loss_chgden.data)
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret
    
    # Get an image of current generation (especially for vq-vae training)
    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_isosurface(self.renderer, self.gen_df)
            
        vis_tensor_names = [
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    # save model along with optimizer and scheduler
    def save(self, label, global_step, save_opt=False):

        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'df': self.df_module.state_dict(),
            'opt': self.optimizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()
        
        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    # load model from the checkpoint file
    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.df.load_state_dict(state_dict['df'])
        
        if 'vqvae' in state_dict.keys():
            self.vqvae.load_state_dict(state_dict['vqvae'])
        elif 'vqvae_tot' in state_dict.keys(): # in the early stages of development, models referred to as vqvae_tot
            self.vqvae.load_state_dict(state_dict['vqvae_tot'])
        else:
            raise ValueError("saved vqvae model is not recognized.")
          
        if self.isTrain:
            self.start_i = state_dict['global_step']
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
                
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))