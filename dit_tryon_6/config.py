config = {
    "img_size": [
        512,
        384
    ],
    "batch_size": 4,
    "lr": 0.0003,
    "dim": 768,
    "k": 64,
    "patch_size": 16,
    "pose_dim": 150,
    "depth": 16,
    "heads": 8,
    "mlp_dim": 3072,
    "fid_batch_size": 4,
    "gen_batch_size": 4,
    "steps": 128,
    "sampling_steps": 128,
    "warmup_steps": 2000,
    "ema_decay": 0.999,
    "n_fid_real": 4,
    "n_fid_gen": 4,
    "n_iter": 10000000,
    "plot_shape": [2,2],
    "seed": 42,
    "cond_dropout_prob": 0.1,
    "guidance_scale": 3.0,
    "fp16": False
}