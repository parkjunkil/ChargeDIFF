# Reference: The model architecture is adapted from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

import os
from collections import OrderedDict
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm
import torch
from torch import nn, optim
from models.base_model import BaseModel
from models.networks.vqvae_networks.network import VQVAE
from models.losses import VQLoss
import utils.util
from utils.util_3d import init_mesh_renderer, render_isosurface
from utils.distributed import reduce_loss_dict

class VQVAEModel(BaseModel):
    def name(self):
        return 'VQVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        # -------------------------------
        # Define Networks
        # -------------------------------

        # model
        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig

        self.vqvae = VQVAE(ddconfig, n_embed, embed_dim)
        self.vqvae.to(self.device)

        if self.isTrain:
            # define loss functions
            codebook_weight = configs.lossconfig.params.codebook_weight
            self.loss_vq = VQLoss(codebook_weight=codebook_weight).to(self.device)

            # initialize optimizers
            self.optimizer = optim.Adam(self.vqvae.parameters(), lr=opt.lr, betas=(0.5, 0.9),foreach=False)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=100)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # continue training
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        # setup renderer
        dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for saving best ckpt
        self.best_iou = -1e12

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.vqvae_module = self.vqvae.module
        else:
            self.vqvae_module = self.vqvae

        #for resume
        self.start_i = 0

    def switch_eval(self):
        self.vqvae.eval()
        
    def switch_train(self):
        self.vqvae.train()

    # for multi-gpu training
    def make_distributed(self, opt):
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        
    # load batch and assign charge density
    def set_input(self, input):
        x = input['chgden']
        self.x = x
        self.cur_bs = x.shape[0] # to handle last batch

        vars_list = ['x']

        self.tocuda(var_names=vars_list)


    def forward(self):
        self.x_recon, self.qloss = self.vqvae(self.x, verbose=False)


    @torch.no_grad()
    def inference(self, data, should_render=False, verbose=False):
        self.switch_eval()
        self.set_input(data)

        with torch.no_grad():
            self.z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)
            self.x_recon = self.vqvae_module.decode_no_quant(self.z)

        self.switch_train()

    # iou is used as a metric to assess the performance of vq-vae
    def test_iou(self, data, thres=0.5):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        self.inference(data, should_render=False)

        x = self.x
        x_recon = self.x_recon
        iou = utils.util.iou(x, x_recon, thres)

        return iou
    
    # obtain loss for validation set
    def val_loss(self, val_dl):
        
        tot_loss = 0
        self.switch_eval()
        
        with torch.no_grad():
            for ix, val_data in tqdm(enumerate(val_dl), total=len(val_dl)):
                self.set_input(val_data)
                self.inference(val_data, should_render=False)
                loss, _ = self.loss_vq(self.qloss, self.x, self.x_recon)
                tot_loss += loss
 
        self.switch_train()

        return tot_loss / len(val_dl)

    # obtain loss metrics during the evaluation phase
    def eval_metrics(self, dataloader, thres=0.5, global_step=0):
        self.switch_eval()
        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()
        
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name, global_step) # pass 0 just now

        self.switch_train()
        return ret

    # back-propagation for model training
    def backward(self):
        total_loss, loss_dict = self.loss_vq(self.qloss, self.x, self.x_recon)
        self.loss = total_loss
        self.loss_dict = reduce_loss_dict(loss_dict)
        self.loss_total = loss_dict['loss_total']
        self.loss_codebook = loss_dict['loss_codebook']
        self.loss_nll = loss_dict['loss_nll']
        self.loss_rec = loss_dict['loss_rec']

        self.loss.backward()

    def optimize_parameters(self, total_steps):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
        
    # Get an image of current generation (especially for vq-vae training)
    def get_current_errors(self):
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('codebook', self.loss_codebook.mean().data),
            ('nll', self.loss_nll.mean().data),
            ('rec', self.loss_rec.mean().data),
        ])
        return ret
    
    # obtain isosurface image of charge density with in batch, and its reconstructed output
    def get_current_visuals(self):
        with torch.no_grad():
            self.image = render_isosurface(self.renderer, self.x)
            self.image_recon = render_isosurface(self.renderer, self.x_recon)
        vis_tensor_names = [
            'image',
            'image_recon',
        ]
        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)
    
    # save model along with optimizer and scheduler
    def save(self, label, global_step=0, save_opt=False):
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'opt': self.optimizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': global_step,
        }
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    # retrieve codebook embedding weights
    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret
    
    # load model from the checkpoint file
    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        self.vqvae.load_state_dict(state_dict['vqvae'])
        
        if self.isTrain:
            self.start_i = state_dict['global_step']
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


