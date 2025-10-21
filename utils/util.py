# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

from __future__ import print_function

import os
import random

import numpy as np
from PIL import Image
from einops import rearrange

import torch
from torch import nn
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

import math

################# Apply Early Stopping During Training Phase#################
## Monitor the validation loss and terminate the training phase if there' no progress for longer than the patience

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        #EarlyStopping Class Initiate
        #param patience: number of checkpoints to wait
        #param delta: minimum progress of loss to admit
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset the counter as there was progress
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Set flag for termination

    def reset(self):
        # Rest Early Stopping
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

################# START: PyTorch Tensor functions #################

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
        # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid( image_tensor, nrow=4 )

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = ( np.transpose( image_numpy, (1, 2, 0) ) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)

def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    # assert tensor.dim() == 3
    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

################# END: PyTorch Tensor functions #################


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
'''
def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.
    x_gt_mask[x_gt <= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b c d h w -> b (c d h w)')
    union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou
'''

def iou(x_gt, x, thres=0.6):

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres] = 0.
    x_gt_mask[x_gt <= thres] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b c d h w -> b (c d h w)')
    union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou

#################### START: MISCELLANEOUS ####################
def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

#################### END: MISCELLANEOUS ####################



# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):
	
	def __init__(self, optimizer, warmup_steps):
		self.warmup_steps = warmup_steps
		super().__init__(optimizer)

	def get_lr(self):
		last_epoch = max(1, self.last_epoch)
		scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
		return [base_lr * scale for base_lr in self.base_lrs]


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class NoiseLevelEncoding(torch.nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        device = t.device
        self.div_term = self.div_term.to(device)
   
        x = torch.zeros((t.shape[0], self.d_model), device=device)
        
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])
        return self.dropout(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def standard_beta_schedule(timesteps, scale = 1.0):
    return torch.tensor([1 / (scale * timesteps - x) for x in range(0, timesteps)])

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal(x, sigma, N, T)

def sigma_norm(sigma, T=1.0, sn = 10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T)
    return (normal_ ** 2).mean(dim = 0)


class BetaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        scheduler_mode,
        device,
        beta_start = 0.0001,
        beta_end = 0.02
    ):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'standard':
            betas = standard_beta_schedule(timesteps)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        #alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        sigmas = torch.zeros_like(betas)
        sigmas[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas = torch.sqrt(sigmas)
        
        self.betas =  betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sigmas = sigmas.to(device)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)


class SigmaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        device,
        sigma_begin = 0.01,
        sigma_end = 1.0
    ):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))
        sigmas_norm_ = sigma_norm(sigmas)

        sigmas = torch.cat([torch.zeros([1]), sigmas], dim=0)
        sigmas_norm_ = torch.cat([torch.ones([1]), sigmas_norm_], dim=0)

        self.sigmas = sigmas.to(device)
        self.sigmas_norm = sigmas_norm_.to(device)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)
    
    
# D3PM for atom types
class D3PM(nn.Module):
    def __init__(
        self,
        beta_scheduler,
        num_timesteps,
        max_atoms,
        d3pm_hybrid_coeff,
        device,
    ):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.num_timesteps = num_timesteps
        self.max_atoms = max_atoms
        self.hybrid_coeff = d3pm_hybrid_coeff
        self.eps = 1.0e-6

        # transition matrix for absorbing
        q_one_step_mats = torch.stack(
            [
                self.get_absorbing_transition_mat(t, device)
                for t in range(0, self.num_timesteps+1)
            ],
            dim=0,
        )
        self.register_buffer("q_one_step_mats", q_one_step_mats)

        # construct transition matrices for q(x_t | x_0)
        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps+1):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.num_timesteps+1,
            self.max_atoms,
            self.max_atoms,
        )

        self.q_one_step_transposed = self.q_one_step_mats.transpose(1, 2)

    def get_absorbing_transition_mat(self, t: int, device):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Args:
            t (int): timestep.
            max_atoms (int): maximum number of atoms (103 + 1 for dummy atom).
                Defaults to 104.

        Returns:
            Q_t: transition matrix. shape = (max_atoms, max_atoms)
        """
        # get beta at timestep t
        beta_t = self.beta_scheduler.betas[t]

        diag = torch.full((self.max_atoms,), 1 - beta_t)
        mat = torch.diag(diag, 0).to(device)
        # add beta_t at first row
        
        mat[:, 0] += beta_t
        return mat

    def at(
        self,
        a: torch.Tensor,
        t_per_node: torch.Tensor,
        x: torch.Tensor,
    ):
        """Extract coefficients at specified timesteps t - 1 and conditioning data x.
 
        Args:
            a (torch.Tensor): matrix of coefficients. [num_timesteps, max_atoms, max_atoms]
            t_per_node (torch.Tensor): timesteps.[B_n]
            x (torch.Tensor): atom_types. [B_n]

        Returns:
            a[t, x] (torch.Tensor): coefficients at timesteps t and data x. [B_n, max_atoms]
        """
        a = a.to(x.device)
        bs = t_per_node.shape[0]
        t_per_node = t_per_node.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t_per_node-1, x, :]

    def q_sample(
        self,
        x_0: torch.Tensor,
        t_per_node: torch.Tensor,
        noise: torch.Tensor,
    ):
        """Sample from q(x_t | x_0) (i.e. add noise to the data).
        q(x_t | x_0) = Categorical(x_t ; p = x_0 Q_{1...t})

        Args:
            x_0 (torch.Tensor): Image data at t=0. [B, C, H, W]
            t_per_node (torch.Tensor): Timesteps. [B_n]
            noise (torch.Tensor): Noise. [B_n, max_atoms]

        Returns:
            torch.Tensor: [B_n, max_atoms]
        """
        logits = torch.log(self.at(self.q_mats, t_per_node, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t_per_node: torch.Tensor,
        is_x_0_one_hot: bool = False,
    ):
        """Compute logits for q(x_{t-1} | x_t, x_0)."""
        if is_x_0_one_hot:
            x_0_logits = x_0.clone()
        else:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.max_atoms) + self.eps
            )

        assert x_0_logits.shape == x_t.shape + (self.max_atoms,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        fact1 = self.at(self.q_one_step_transposed, t_per_node, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)
        qmats2 = self.q_mats[t_per_node - 2]
        fact2 = torch.einsum("b...c, bcd -> b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t_per_node.reshape((t_per_node.shape[0], *[1] * (x_t.dim())))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def categorical_kl_logits(self, logits1, logits2, eps=1.0e-6):
        """KL divergence between categorical distributions.

        Distributions parameterized by logits.

        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.

        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = torch.softmax(logits1 + eps, dim=-1) * (
            torch.log_softmax(logits1 + eps, dim=-1)
            - torch.log_softmax(logits2 + eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def p_logits(
        self,
        pred_x_start_logits: torch.Tensor,
        x_t_atom_types: torch.Tensor,
        t_per_node: torch.Tensor,
        noise: torch.Tensor,
    ):
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t_atom_types, t_per_node, is_x_0_one_hot=True
        )

        noise = torch.clamp(noise, min=self.eps, max=1.0)
        # if t == 1, use x_0_logits
        nonzero_mask = (
            (t_per_node != 1)
            .to(x_t_atom_types.dtype)
            .view(-1, *([1] * (x_t_atom_types.ndim)))
        )
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * nonzero_mask, dim=-1
        )
        return sample