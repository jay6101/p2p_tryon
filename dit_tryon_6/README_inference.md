# DiT Virtual Try-on Inference

This script allows you to run inference on a trained DiT virtual try-on model to generate synthesized try-on images.

## Overview

The inference script loads a trained model and processes a sample from your dataset to generate a virtual try-on result. It will save all relevant images:

- **Original**: The original target person image
- **Cloth Agnostic**: The person without clothes (body shape + pose)
- **Garment**: The garment to be tried on
- **Synthesized**: The generated try-on result
- **Comparison**: A side-by-side grid showing all four images

## Usage

### Command Line Interface

Basic usage:
```bash
python inference.py --data_dir /path/to/your/dataset
```

Full options:
```bash
python inference.py \
    --model_dir model_4 \
    --data_dir /path/to/your/dataset \
    --output_dir inference_output \
    --sample_idx 0 \
    --guidance_scale 3.0 \
    --sampling_steps 100 \
    --seed 42 \
    --device cuda:1
```

### Programmatic Usage

```python
from inference import run_inference

# Run inference on a specific sample
result = run_inference(
    model_dir='model_4',
    data_dir='/path/to/your/dataset',
    output_dir='my_results',
    sample_idx=5,
    guidance_scale=3.0,
    sampling_steps=100,
    seed=42,
    device='cuda:1'
)
```

### Batch Processing Example

Use the provided example script to process multiple samples:

```bash
# Edit the data_dir path in run_inference_example.py first
python run_inference_example.py
```

## Parameters

- `--model_dir`: Directory containing the trained model checkpoint (default: `model_4`)
- `--data_dir`: **Required** - Directory containing your dataset
- `--output_dir`: Where to save inference results (default: `inference_output`)
- `--sample_idx`: Index of the sample to process (default: `0`)
- `--guidance_scale`: Scale for classifier-free guidance. Higher values = stronger conditioning (default: `3.0`)
- `--sampling_steps`: Number of diffusion sampling steps. More steps = better quality but slower (default: `100`)
- `--seed`: Random seed for reproducible generation (default: `42`)
- `--device`: Device to use for inference (default: `cuda:1`)

## Dataset Structure

Your dataset should follow this structure:
```
dataset/
├── sample_001/
│   ├── 1.jpg                    # Target person image
│   ├── 2.jpg                    # Garment image
│   ├── 1_pose.jpg               # Target person pose
│   ├── 2_pose.jpg               # Garment pose
│   ├── 1/alpha/1_new.png        # Cloth agnostic mask
│   └── 2/alpha/1.png            # Garment mask
├── sample_002/
│   └── ... (same structure)
└── ...
```

## Output Structure

For each sample, the script creates:
```
output_dir/
├── {sample_id}_original.png      # Original target image
├── {sample_id}_cloth_agnostic.png # Person without clothes
├── {sample_id}_garment.png       # Garment to try on
├── {sample_id}_synthesized.png   # Generated result
└── {sample_id}_comparison.png    # Side-by-side comparison
```

## Model Requirements

- The script expects a trained model checkpoint at `{model_dir}/last_ckpt.pt`
- The checkpoint should contain `ema` (Exponential Moving Average) weights for best results
- Model configuration is loaded from `{model_dir}/config.json`

## Performance Tips

- **Guidance Scale**: Try values between 1.0-5.0. Higher values give stronger conditioning but may reduce diversity
- **Sampling Steps**: 50 steps for quick results, 100+ for best quality
- **Device**: Use GPU for faster inference (`cuda:0`, `cuda:1`, etc.)
- **Batch Processing**: Process multiple samples by varying `sample_idx`

## Troubleshooting

1. **CUDA out of memory**: Reduce `sampling_steps` or use CPU device
2. **Sample index out of range**: Check your dataset size
3. **Missing checkpoint**: Ensure `last_ckpt.pt` exists in model directory
4. **Import errors**: Make sure all required modules (`dit_4.py`, `image_datasets.py`, etc.) are in the same directory

## Example Results

The script will output something like:
```
Loading checkpoint from: model_4/last_ckpt.pt
Loaded EMA model weights
Processing sample: sample_001
Input shapes:
  Target image: torch.Size([1, 3, 512, 384])
  Cloth agnostic: torch.Size([1, 6, 512, 384])
  Garment: torch.Size([1, 6, 512, 384])
Running inference...
Generated image shape: torch.Size([1, 3, 512, 384])
Saved images:
  Original: inference_output/sample_001_original.png
  Cloth agnostic: inference_output/sample_001_cloth_agnostic.png
  Garment: inference_output/sample_001_garment.png
  Synthesized: inference_output/sample_001_synthesized.png
Comparison grid saved to: inference_output/sample_001_comparison.png
Inference completed successfully! 