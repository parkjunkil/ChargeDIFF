# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion
# Code for the training of various models (e.g., VQ-VAE, MOF-Constructor, Unconditional & Conditional MOFFUSION)

import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn

from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from utils.util import EarlyStopping

from utils.distributed import get_rank


import torch
from utils.visualizer import Visualizer

def train_main_worker(opt, model, train_dl, val_dl, test_dl, visualizer, device):

    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_dl)
    #val_dg = get_data_generator(val_dl)
    test_dg = get_data_generator(test_dl)

    
    pbar = tqdm(total=opt.total_iters)

    early_stopping = EarlyStopping(patience=opt.patience)

    iter_start_time = time.time()
    for iter_i in range(opt.total_iters):

        torch.cuda.empty_cache()

        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()
        
        data = next(train_dg)

        model.set_input(data)
        
        model.optimize_parameters(iter_i)

        if get_rank() == 0:
            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()

                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(iter_i, errors, t)

            
            # display charge density during training (only for vqvae training)
            if 'vqvae' in opt.model and iter_i % opt.display_freq == 0:
                if iter_i == 0 and opt.debug == "1":
                    pbar.update(1)
                    continue

                # eval
                model.inference(data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='train')
                
                # test
                test_data = next(test_dg)
                model.inference(test_data)
                visualizer.display_current_results(model.get_current_visuals(), iter_i, phase='test')
                
            # save checkpoints during the training
            if iter_ip1 % opt.save_latest_freq == 0:
                cprint('saving the latest model (current_iter %d)' % (iter_i), 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)

            if iter_ip1 % opt.save_steps_freq == 0:
                cprint('saving the model at iters %d' % iter_ip1, 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)
                cur_name = f'steps-{iter_ip1}'
                model.save(cur_name, iter_ip1)


            # save and print loss metrics
            if iter_ip1 % opt.save_steps_freq == 0:
                
                val_metrics = model.eval_metrics(val_dl)
                test_metrics = model.eval_metrics(test_dl)
                
                visualizer.print_current_metrics(iter_ip1, val_metrics, phase='val')
                visualizer.print_current_metrics(iter_ip1, test_metrics, phase='test')
                
                cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                    (
                        iter_ip1,
                        time.time() - iter_start_time,
                        os.path.abspath( os.path.join(opt.logs_dir, opt.name) )
                    ), 'blue', attrs=['bold']
                    )
            
                
            # check early stopping criterion
            if iter_ip1 % len(train_dl) == 0:
                
                if 'vqvae' in opt.model:
                    val_loss = model.val_loss(val_dl)
                else:
                    val_metrics = model.eval_metrics(val_dl)
                    print(val_metrics)
                    val_loss = val_metrics['loss_tot']
                    
                early_stopping(val_loss)
  
                model.update_learning_rate(val_loss)

            if early_stopping.early_stop:
                epoch = iter_ip1 // len(train_dl)
                print(f"Early stopping at epoch {epoch + 1}")
                break

        pbar.update(1)
        
        

if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()

    device = opt.device
    rank = opt.rank

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    train_dl, val_dl, test_dl = CreateDataLoader(opt)
    train_ds, val_ds, test_ds = train_dl.dataset, val_dl.dataset, test_dl.dataset

    dataset_size = len(train_ds)


    # main loop
    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    
    # visualizer
    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        dset_f = inspect.getfile(train_ds.__class__)
        cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
        os.system(f'cp {model_f} {modelf_out}')
        os.system(f'cp {dset_f} {dsetf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')
            
        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')


    train_main_worker(opt, model, train_dl, val_dl, test_dl, visualizer, device)
    
