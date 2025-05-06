# minDiT hyperparameters
config = {  
    'img_size': (512,384),
    'batch_size': 8,
    'lr': 0.0001,
    'dim': 768,
    'k': 64, # linformer dim
    'patch_size': 16,
    'pose_dim': 150,
    'depth': 16, 
    'heads': 8, 
    'mlp_dim': 768*4, 
    'fid_batch_size': 4, # inception 
    'gen_batch_size': 4,
    'steps': 100,  # eval diff steps
    'warmup_steps': 10000, # add this entry
    'ema_decay': 0.999,
    'n_fid_real': 4,
    'n_fid_gen': 4,
    'n_iter': 10000000,
    'plot_shape': (2,2),
    'seed': 42
}
