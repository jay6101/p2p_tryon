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
from dit import DiT
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

    # dataset
    train_loader = create_tryon_loader(
        data_dir, img_size, batch_size
    )
    
    # model
    model = DiT(img_size, dim, patch_size,
            depth, heads, mlp_dim, k, pose_dim=pose_dim)
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
        garment_pose = batch_data['garment_pose'].to(device)
        target_pose = batch_data['target_pose'].to(device)
        
        # Apply diffusion
        xt, t, target = diffusion.diffuse(target_image)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
    
        # Forward + backward + optimize
        outputs = model(xt, t, cloth_agnostic, garment, garment_pose, target_pose)
        weight  = (t**2 + diffusion.sigma_data**2)/(t**2 * diffusion.sigma_data**2)
        loss = (weight * ((outputs - target).square()).mean(dim=(1,2,3))).mean()
        # loss = loss_fn(outputs, target)
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
            # Generate a batch of samples
            gen_batch = sample_with_conditions(
                diffusion, ema, batch_data, sz, steps=steps, seed=seed
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
            gen_batches_for_fid(
                diffusion, ema, train_loader, n_fid_real, gen_batch_size, 
                steps, gen_dir, img_size
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
                'train_loss': train_loss
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
    garment_pose = batch_data['garment_pose'][0:sz[0]].to(device)
    target_pose = batch_data['target_pose'][0:sz[0]].to(device)
    
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
                    x, t_steps, i, model, cloth_agnostic, garment, garment_pose, target_pose
                )
    else:
        x = torch.randn(sz, device=device) * sigma_max
        t_steps = get_sigmas_karras(steps, device=device, sigma_max=sigma_max)
        
        for i in range(len(t_steps) - 1):
            x = diffusion.edm_sampler_with_conditions(
                x, t_steps, i, model, cloth_agnostic, garment, garment_pose, target_pose
            )
            
    return x.cpu()

def gen_batches_for_fid(diffusion, model, data_loader, n_images, batch_size, 
                steps, dir_path, img_size):
    """
    Generate batches for FID evaluation using the conditions from the dataset
    """
    n_batches = n_images // batch_size
    n_used_imgs = n_batches * batch_size
    
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        
        # Get a batch of data
        batch_data = next(data_loader)
        
        # Generate a batch with these conditions
        gen_batch = sample_with_conditions(
            diffusion, model, batch_data, 
            (batch_size, 3, img_size[0], img_size[1]), 
            steps=steps
        )
        
        # Denormalize
        gen_batch = (gen_batch + 1.) / 2
        
        # Save images
        img_index = start
        for img in gen_batch:
            file_name = os.path.join(dir_path, f'{str(img_index)}.png')
            save_image(img, file_name)
            img_index += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model_1')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fid_real_dir', type=str, default=None)
    parser.add_argument('--iter_interval', type=int, default=1000)
    parser.add_argument('--fid_interval', type=int, default=5000)
    args = parser.parse_args()

    conf = Config(config, args.model_dir)
    train(
        args.model_dir, args.data_dir, args.fid_real_dir, 
        args.iter_interval, args.fid_interval, conf
    )


if __name__ == '__main__':
    main()
