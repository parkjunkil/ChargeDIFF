import click
import time
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData

@click.command()
@click.option('--model', default='uncond', help="ChargeDIFF model type in [uncond, magnetic_density, bandgap, crystal_density].")
@click.option('--batch-size', default=10, help="Number of structures per batch.")
@click.option('--num-batch', default=2, help="Number of batches to generate.")
@click.option('--target', default=None, help="Target for conditional generation")
@click.option('--save-dir', default='./sample/uncond', help="Directory to save the results.")

def generate_structures(model, batch_size, num_batch, target, save_dir):

    from utils.demo_util import ChargeDIFFOpt
    from models.base_model import create_model
    
    seed = 42

    opt = ChargeDIFFOpt(gpu_ids=gpu_ids, seed=seed)

    opt.init_dset_args()
    opt.init_model_args()
        
    opt.vq_cfg="./configs/vqvae.yaml"
    opt.vq_ckpt="./saved_ckpt/vqvae.pth"
        
    assert model in [None, 'uncond', 'magnetic_density', 'bandgap', 'crystal_density'], \
    f"Unexpected model value '{model}' provided. Please choose from 'uncond', 'magnetic_density', 'bandgap', or 'crystal_density'."

        
    if model == None or model =='uncond':
        opt.model='chargediff_uncond'
        opt.df_cfg = './configs/chargediff_uncond.yaml'
        opt.ckpt = './saved_ckpt/uncond_64.pth'
        
        ChargeDIFF = create_model(opt)
        cprint(f'[*] "{ChargeDIFF.name()}" loaded.', 'cyan')
        
        for i in range(num_batch):
            gen_crys, chgden_tot = ChargeDIFF.ab_initio_gen(num_batch=1, batch_size=batch_size, dataset='mp_20_charge', cutoff=0.5, step_lr=1e-5)

            for j in range(len(gen_crys)):
                cry = gen_crys[j]
                chgden = chgden_tot[j]
                structure = cry.structure

                volumetric = VolumetricData(structure=structure, data={'total': chgden})

                idx = i * batch_size + j
                volumetric.write_file(f'{save_dir}/{idx}.vasp')
                structure.to_file(f'{save_dir}/{idx}.cif')
        
    else:
        assert target is not None, \
        f"Unexpected target value '{target}' provided. Please give numerical target value for conditional generation."
        
        if isinstance(target, str):
            target = float(target)

        
        opt.df_cfg = './configs/chargediff_cond.yaml'
    
        if model == 'magnetic_density':
            opt.model='chargediff_magnetic_density'
            opt.ckpt = './saved_ckpt/magden_64.pth'
        
        elif model == 'bandgap':
            opt.model='chargediff_bandgap'
            opt.ckpt = './saved_ckpt/bandgap_64.pth'
        
        elif model == 'crystal_density':
            opt.model='chargediff_density'
            opt.ckpt = './saved_ckpt/density_64.pth'
            
        from datasets.base_dataset import CreateDataset
        opt.dataset_mode = 'MP-20-Charge'
        train_dataset, val_dataset, test_dataset = CreateDataset(opt)
        scaler = train_dataset.scaler

                
        ChargeDIFF = create_model(opt)
        cprint(f'[*] "{ChargeDIFF.name()}" loaded.', 'cyan')
        
        for i in range(num_batch):
            
            gen_crys, chgden_tot = ChargeDIFF.cond(batch_size = batch_size, scaler=scaler, target=target, dataset='mp_20_charge', cutoff = 0.5)


            for j in range(len(gen_crys)):
                cry = gen_crys[j]
                chgden = chgden_tot[j]
                structure = cry.structure
                
                volumetric = VolumetricData(structure = structure, data={'total': chgden})

                idx = i*batch_size + j
                
                volumetric.write_file(f'{save_dir}/{idx}.vasp')
                structure.to_file(f'{save_dir}/{idx}.cif')
        

if __name__ == "__main__":
    import os
    gpu_ids = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
    
    from termcolor import  cprint

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    generate_structures()
