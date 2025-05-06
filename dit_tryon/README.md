# Diffusion Transformer for Virtual Try-On

## Dependencies
- Python 3.9
- Pytorch 2.1.1


## Virtual Try-On DiT Architecture
This implementation extends the original DiT architecture for virtual try-on tasks with multiple conditions:

1. **Cloth-agnostic Image**: Person image with clothing region masked out
2. **Garment Image**: The garment to be virtually tried on
3. **Garment Pose**: 150-dimensional pose vector for the garment
4. **Target Pose**: 150-dimensional pose vector for the desired pose

The architecture has been modified to:
- Tokenize both cloth-agnostic and garment images using the same encoder
- In each DiT block:
  - Concatenate cloth-agnostic tokens (detached from graph) with noisy tokens for self-attention
  - Add cross-attention where queries are noisy tokens and keys/values are garment + pose tokens
  - Process pose vectors through MLPs in each block

## Dataset Structure
The dataset should be organized as follows:
```
data_dir/
├── sample_1/
│   ├── target.jpg       # Target image (ground truth)
│   ├── agnostic.jpg     # Cloth-agnostic image
│   ├── garment.jpg      # Garment image
│   └── meta.json        # Contains 'garment_pose' and 'target_pose' vectors
├── sample_2/
│   ├── ...
...
```

Alternatively, you can provide a `metadata.json` file in the data directory with the paths to all components.

## Training Diffusion Transformer
Use `--data_dir=<data_dir>` to specify the dataset path.
```
python train.py --data_dir=./data/
```


## Samples
Sample output from minDiT (39.89M parameters) on CIFAR-10:

<img src="./images/diff_cifar.png" width="550px"></img>

Sample output from minDiT on CelebA:

<img src="./images/diff_celeba64.png" width="650px"></img>

More samples:

<img src="./images/mindit_cifar.gif" width="650px"></img>
<img src="./images/mindit_celeba64.gif" width="550px"></img>

## Hparams setting
Adjust hyperparameters in the `config.py` file. For virtual try-on, key parameters include:
- `pose_dim`: Dimension of pose vectors (default: 150)
- `img_size`: Size of input images (default: 512)
- `dim`: Embedding dimension (default: 384)

Implementation notes:
- minDiT is designed to offer reasonable performance using a single GPU (RTX 3080 TI).
- minDiT largely follows the original DiT model, with modifications for virtual try-on.
- DiT Block with adaLN-Zero and additional cross-attention for garment and pose conditions.
- [EDM](https://arxiv.org/abs/2206.00364) sampler.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.


## todo
- Add Classifier-Free Diffusion Guidance and conditional pipeline.
- Add Latent Diffusion and Autoencoder training.
- Add generate.py file.


## Licence
MIT
