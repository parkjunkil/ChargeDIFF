# Reference: The code is adapted from the LDM repo: https://github.com/yccyenchicheng/SDFusion

from termcolor import colored
import torch

from models.networks.vqvae_networks.network import VQVAE

# to load saved vqvae and use if for chargediff training
def load_vqvae(vq_conf, vq_ckpt, opt=None):
    assert type(vq_ckpt) == str

    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig

    vqvae = VQVAE(ddconfig, n_embed, embed_dim)
    
    map_fn = lambda storage, loc: storage
    state_dict = torch.load(vq_ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.device)
    vqvae.eval()
    return vqvae