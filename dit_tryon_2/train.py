'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Dec 2023
Modified for Virtual Try-on with multiple conditions
'''
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import warnings
from copy import deepcopy
from collections import OrderedDict
import argparse
from fid import get_fid
from image_datasets import create_tryon_loader
from config import config
from dit_5 import DiT
from utils import * 
from diff_utils import *
from tqdm import tqdm, trange

warnings.filterwarnings("ignore")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data.float(), alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def train(model_dir, data_dir, fid_real_dir, 
          iter_interval, fid_interval, conf):
    if fid_real_dir == None:
        fid_real_dir = data_dir
    img_size = conf.img_size
    batch_size = conf.batch_size
    lr = conf.lr
    dim = conf.dim
    ema_decay = conf.ema_decay
    patch_size = conf.patch_size
    depth = conf.depth
    heads = conf.heads
    mlp_dim = conf.mlp_dim
    k = conf.k
    pose_dim = conf.pose_dim
    fid_batch_size = conf.fid_batch_size
    gen_batch_size = conf.gen_batch_size
    steps = conf.steps
    n_fid_real = conf.n_fid_real
    n_fid_gen = conf.n_fid_gen
    n_iter = conf.n_iter
    plot_shape = conf.plot_shape
    seed = conf.seed
    
    # Classifier-free guidance parameters
    cond_dropout_prob = conf.cond_dropout_prob if hasattr(conf, 'cond_dropout_prob') else 0.1
    guidance_scale = conf.guidance_scale if hasattr(conf, 'guidance_scale') else 3.0

    # dataset
    train_loader = create_tryon_loader(
        data_dir, img_size, batch_size
    )
    
    # model
    model = DiT(img_size, dim, patch_size,
            depth, heads, mlp_dim, in_channels=3)
    diffusion = Diffusion()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.995), eps=1e-8, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=conf.warmup_steps)
    loss_fn = torch.nn.MSELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # create ema
    ema = deepcopy(model).to(device)  
    requires_grad(ema, False)
    
    # logs and ckpt config
    gen_dir = os.path.join(model_dir, 'fid')
    log_img_dir = os.path.join(model_dir, 'log_img')
    log_dir = os.path.join(model_dir, 'log_dir')
    writer = SummaryWriter(log_dir)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    last_ckpt = os.path.join(model_dir, './last_ckpt.pt')
    best_ckpt = os.path.join(model_dir, './best_ckpt.pt')
    
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt)
        start_iter = ckpt['iter'] + 1 # start from iter + 1
        best_fid = ckpt['best_fid']
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        print(f'Checkpoint restored at iter {start_iter}; ' 
                f'best FID: {best_fid}')
    else:
        start_iter = 1
        best_fid = 1000. # init with big value
        print(f'New model')

    # plot shape
    sz = (plot_shape[0] * plot_shape[1], 3, img_size[0], img_size[1])

    # train
    pbar = trange(n_iter, desc="Training", leave=True)
    start = time.time()
    train_loss = 0.0
    update_ema(ema, model, decay=ema_decay) 
    model.train()
    ema.eval()  # EMA model should always be in eval mode
    for idx in pbar:
        i = idx + start_iter
        batch_data = next(train_loader)
        
        # Unpack the batch data
        target_image = batch_data['target_image'].to(device)
        cloth_agnostic = batch_data['cloth_agnostic'].to(device)
        garment = batch_data['garment'].to(device)
        
        # Apply diffusion
        xt, t, target = diffusion.diffuse(target_image)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
    
        # Apply classifier-free guidance (conditioning dropout)
        # Create empty/null conditioning inputs
        B = target_image.shape[0]
        
        # Create dropout mask - True means keep conditioning, False means drop
        use_cond_mask = torch.rand(B, device=device) >= cond_dropout_prob
        
        # Create null/empty conditioning tensors of the same shape
        null_cloth_agnostic = torch.zeros_like(cloth_agnostic)
        null_garment = torch.zeros_like(garment)
        
        # Apply dropout to conditioning based on mask
        # For each sample in batch, either keep or replace with null conditioning
        cloth_agnostic_input = torch.where(use_cond_mask.view(B, 1, 1, 1), cloth_agnostic, null_cloth_agnostic)
        garment_input = torch.where(use_cond_mask.view(B, 1, 1, 1), garment, null_garment)
        
        # Forward pass with randomly dropped conditioning
        outputs = model(xt, t, cloth_agnostic_input, garment_input)
        
        # Apply loss weighting based on timestep
        weight = (t**2 + diffusion.sigma_data**2)/(t**2 * diffusion.sigma_data**2)
        loss = (weight * ((outputs - target).square()).mean(dim=(1,2,3))).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        sched.step()  
        update_ema(ema, model)
        train_loss += loss.item()
        pbar.set_postfix({
        "iter": i,
        "loss": f"{loss.item():.4f}",
        # optionally smoothed: "avg_loss": f"{(train_loss/(idx+1)):.4f}"
        })

        if i % iter_interval == 0:
            # plot
            # Generate a batch of samples with CFG
            gen_batch = sample_with_conditions_cfg(
                diffusion, ema, batch_data, sz, guidance_scale=guidance_scale, steps=steps, seed=seed
            )
            plot_path = os.path.join(log_img_dir, f'{i:04d}.png')
            plot_batch(deprocess(gen_batch), batch_data, plot_shape, plot_path, img_size=img_size)
            
            # metrics
            train_loss /= iter_interval
            print(f'Time for iter {i} is {time.time()-start:.4f}'
                        f'sec Train loss: {train_loss:.4f}')
            writer.add_scalar('train_loss_iter', train_loss, i)
            writer.add_scalar('train_loss_n_img', train_loss, i * batch_size)
            writer.flush()
            train_loss = 0.0
            start = time.time()
            model.train()

        if i % fid_interval == 0:
            # fid
            print('Generating eval batches...')
            gen_batches_for_fid_cfg(
                diffusion, ema, train_loader, n_fid_real, gen_batch_size, 
                steps, gen_dir, img_size, guidance_scale=guidance_scale
            )
            fid = get_fid(
                fid_real_dir, gen_dir, n_fid_real, n_fid_gen,
                device, fid_batch_size
            )
            print(f'FID: {fid}')
            writer.add_scalar('FID_iter', fid, i)
            writer.add_scalar('FID_n_img', fid, i * batch_size)
            writer.flush()

            # ckpt
            ckpt_data = {
                'iter': i,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': optimizer.state_dict(),
                'fid': fid,
                'best_fid': min(fid, best_fid),
                'train_loss': train_loss,
                'guidance_scale': guidance_scale,
                'cond_dropout_prob': cond_dropout_prob
            }
            
            torch.save(ckpt_data, last_ckpt)
            print(f'Checkpoint saved at iter {i}')
            
            if fid <= best_fid:
                torch.save(ckpt_data, best_ckpt)
                best_fid = fid
                print(f'Best checkpoint saved at iter {i}')
                           
            start = time.time()
            model.train()

@torch.no_grad()
def sample_with_conditions(diffusion, model, batch_data, sz, steps=100, sigma_max=80., seed=None):
    """
    Sample from the model with fixed conditions from a batch
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get a single set of conditions to use for the whole batch
    cloth_agnostic = batch_data['cloth_agnostic'][0:sz[0]].to(device)
    garment = batch_data['garment'][0:sz[0]].to(device)
    
    # Expand conditions to match batch size
    batch_size = sz[0]
    # cloth_agnostic = cloth_agnostic.expand(batch_size, -1, -1, -1)
    # garment = garment.expand(batch_size, -1, -1, -1)
    # garment_pose = garment_pose.expand(batch_size, -1)
    # target_pose = target_pose.expand(batch_size, -1)
    
    # Sample with fixed conditions
    if seed is not None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            x = torch.randn(sz, device=device) * sigma_max
            t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
            
            for i in range(len(t_steps) - 1):
                x = diffusion.edm_sampler_with_conditions(
                    x, t_steps, i, model, cloth_agnostic, garment
                )
    else:
        x = torch.randn(sz, device=device) * sigma_max
        t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
        
        for i in range(len(t_steps) - 1):
            x = diffusion.edm_sampler_with_conditions(
                x, t_steps, i, model, cloth_agnostic, garment
            )
    return x

@torch.no_grad()
def sample_with_conditions_cfg(diffusion, model, batch_data, sz, guidance_scale=3.0, steps=100, sigma_max=80., seed=None):
    """
    Sample from the model with classifier-free guidance
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get conditions to use for the whole batch
    cloth_agnostic = batch_data['cloth_agnostic'][0:sz[0]].to(device)
    garment = batch_data['garment'][0:sz[0]].to(device)
    
    # Create null/empty conditioning tensors
    null_cloth_agnostic = torch.zeros_like(cloth_agnostic)
    null_garment = torch.zeros_like(garment)
    
    # Sample with classifier-free guidance
    if seed is not None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            x = torch.randn(sz, device=device) * sigma_max
            t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
            
            for i in range(len(t_steps) - 1):
                # Get prediction with actual conditioning
                cond_output = diffusion.get_d_with_conditions(
                    model, x, t_steps[i], cloth_agnostic, garment)
                
                # Get prediction with null conditioning
                uncond_output = diffusion.get_d_with_conditions(
                    model, x, t_steps[i], null_cloth_agnostic, null_garment)
                
                # Apply classifier-free guidance
                d = uncond_output + guidance_scale * (cond_output - uncond_output)
                
                # Continue with sampling using the guided prediction
                d_cur = (x - d) / t_steps[i]
                x_next = x + (t_steps[i + 1] - t_steps[i]) * d_cur
                
                if t_steps[i + 1] != 0:
                    # Get prediction with actual conditioning
                    cond_output = diffusion.get_d_with_conditions(
                        model, x_next, t_steps[i + 1], cloth_agnostic, garment)
                    
                    # Get prediction with null conditioning
                    uncond_output = diffusion.get_d_with_conditions(
                        model, x_next, t_steps[i + 1], null_cloth_agnostic, null_garment)
                    
                    # Apply classifier-free guidance
                    d = uncond_output + guidance_scale * (cond_output - uncond_output)
                    
                    d_prime = (x_next - d) / t_steps[i + 1]
                    d_prime = (d_cur + d_prime) / 2
                    x_next = x + (t_steps[i + 1] - t_steps[i]) * d_prime
                
                x = x_next
    else:
        x = torch.randn(sz, device=device) * sigma_max
        t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
        
        for i in range(len(t_steps) - 1):
            # Get prediction with actual conditioning
            cond_output = diffusion.get_d_with_conditions(
                model, x, t_steps[i], cloth_agnostic, garment
            )
            
            # Get prediction with null conditioning
            uncond_output = diffusion.get_d_with_conditions(
                model, x, t_steps[i], null_cloth_agnostic, null_garment
            )
            
            # Apply classifier-free guidance
            d = uncond_output + guidance_scale * (cond_output - uncond_output)
            
            # Continue with sampling using the guided prediction
            d_cur = (x - d) / t_steps[i]
            x_next = x + (t_steps[i + 1] - t_steps[i]) * d_cur
            
            if t_steps[i + 1] != 0:
                # Get prediction with actual conditioning
                cond_output = diffusion.get_d_with_conditions(
                    model, x_next, t_steps[i + 1], cloth_agnostic, garment
                )
                
                # Get prediction with null conditioning
                uncond_output = diffusion.get_d_with_conditions(
                    model, x_next, t_steps[i + 1], null_cloth_agnostic, null_garment
                )
                
                # Apply classifier-free guidance
                d = uncond_output + guidance_scale * (cond_output - uncond_output)
                
                d_prime = (x_next - d) / t_steps[i + 1]
                d_prime = (d_cur + d_prime) / 2
                x_next = x + (t_steps[i + 1] - t_steps[i]) * d_prime
            
            x = x_next
            
    return x

def gen_batches_for_fid(diffusion, model, data_loader, n_images, batch_size, 
                steps, dir_path, img_size):
    """
    Generate batches for FID evaluation
    """
    # Delete existing images if any
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    n_batches = n_images // batch_size
    n_used_imgs = n_batches * batch_size
    sz = (batch_size, 3, img_size[0], img_size[1])
    device = next(model.parameters()).device

    for i in tqdm(range(n_batches)):
        batch_data = next(data_loader)
        gen_batch = sample_with_conditions(diffusion, model, batch_data, sz, steps=steps)
        gen_batch = (gen_batch + 1.) / 2  # Rescale to [0, 1]
        
        img_index = i * batch_size
        for j, img in enumerate(gen_batch):
            file_name = os.path.join(dir_path, f'{str(img_index + j)}.png')
            save_image(img.cpu(), file_name)

def gen_batches_for_fid_cfg(diffusion, model, data_loader, n_images, batch_size, 
                         steps, dir_path, img_size, guidance_scale=3.0):
    """
    Generate batches for FID evaluation using classifier-free guidance
    """
    # Delete existing images if any
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    n_batches = n_images // batch_size
    n_used_imgs = n_batches * batch_size
    sz = (batch_size, 3, img_size[0], img_size[1])
    device = next(model.parameters()).device

    for i in tqdm(range(n_batches)):
        batch_data = next(data_loader)
        gen_batch = sample_with_conditions_cfg(diffusion, model, batch_data, sz, guidance_scale=guidance_scale, steps=steps)
        gen_batch = (gen_batch + 1.) / 2  # Rescale to [0, 1]
        
        img_index = i * batch_size
        for j, img in enumerate(gen_batch):
            file_name = os.path.join(dir_path, f'{str(img_index + j)}.png')
            save_image(img.cpu(), file_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model_4')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fid_real_dir', type=str, default=None)
    parser.add_argument('--iter_interval', type=int, default=1000)
    parser.add_argument('--fid_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--cond_dropout_prob', type=float, default=0.1, 
                        help='Probability of dropping conditioning during training')
    parser.add_argument('--guidance_scale', type=float, default=3.0, 
                        help='Scale for classifier-free guidance during sampling')
    args = parser.parse_args()

    conf = Config(config, args.model_dir)

    train(args.model_dir, args.data_dir, args.fid_real_dir, 
        args.iter_interval, args.fid_interval, conf)

if __name__ == '__main__':
    main()
