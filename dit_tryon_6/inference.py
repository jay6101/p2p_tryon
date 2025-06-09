'''
Inference script for DiT Virtual Try-on model
Loads a sample datapoint and generates synthesized image
'''
import torch
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Import our modules
from image_data_infer import TryonDataset
from config import config
from dit_4 import DiT
from utils import Config, deprocess
from diff_utils import Diffusion, get_sigmas_karras


@torch.no_grad()
def sample_with_conditions_cfg(diffusion, model, cloth_agnostic, garment, 
                               img_size, guidance_scale=3.0, steps=100, sigma_max=80., seed=None):
    """
    Sample from the model with classifier-free guidance
    """
    device = next(model.parameters()).device
    model.eval()
    
    batch_size = cloth_agnostic.shape[0]
    sz = (batch_size, 3, img_size[0], img_size[1])
    
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


def save_image_from_tensor(tensor, path, denormalize=True):
    """
    Save a tensor as an image
    """
    if denormalize:
        # Convert from [-1, 1] to [0, 1] 
        tensor = (tensor + 1.) / 2.
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0., 1.)
    
    # Save the image
    save_image(tensor, path)


def tensor_to_pil(tensor, denormalize=True):
    """
    Convert tensor to PIL Image for display
    """
    if denormalize:
        tensor = (tensor + 1.) / 2.
    tensor = torch.clamp(tensor, 0., 1.)
    
    # Convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    numpy_img = tensor.permute(1, 2, 0).cpu().numpy()
    numpy_img = (numpy_img * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_img)


def create_comparison_grid(target_img, cloth_agnostic, garment, original_garment_person, synthesized, output_path):
    """
    Create a 1x5 grid showing: original, cloth agnostic, garment, original garment person, synthesized
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Convert tensors to PIL images
    target_pil = tensor_to_pil(target_img)
    cloth_agnostic_pil = tensor_to_pil(cloth_agnostic[:3])  # Only RGB channels
    garment_pil = tensor_to_pil(garment[:3])  # Only RGB channels  
    original_garment_person_pil = tensor_to_pil(original_garment_person)
    synthesized_pil = tensor_to_pil(synthesized)
    
    images = [target_pil, cloth_agnostic_pil, garment_pil, original_garment_person_pil, synthesized_pil]
    titles = ['Original', 'Cloth Agnostic', 'Garment', 'Original Garment Person', 'Synthesized']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {output_path}")


def run_inference(model_dir, data_dir, output_dir, sample_idx=0, guidance_scale=3.0, 
                  sampling_steps=100, seed=42, device='cuda:1'):
    """
    Run inference on a single sample
    """
    # Load config
    conf = Config(config, model_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = DiT(
        img_size=conf.img_size, 
        dim=conf.dim, 
        patch_size=conf.patch_size,
        depth=conf.depth, 
        heads=conf.heads, 
        mlp_dim=conf.mlp_dim, 
        k=conf.k
    )
    
    # Load checkpoint
    ckpt_path = os.path.join(model_dir, 'last_ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Load EMA model weights (EMA typically gives better results for inference)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(ckpt['model'])
        print("Loaded regular model weights")
    
    model.to(device)
    model.eval()
    
    # Initialize diffusion
    diffusion = Diffusion()
    
    # Load dataset
    dataset = TryonDataset(data_dir, conf.img_size)
    
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} exceeds dataset size {len(dataset)}")
    
    # Get a sample
    sample_data = dataset[sample_idx]
    sample_id = sample_data['target_sample_id']
    
    print(f"Processing sample: {sample_id}")
    
    # Prepare data
    target_image = sample_data['target_image'].unsqueeze(0).to(device)  # Add batch dimension
    cloth_agnostic = sample_data['cloth_agnostic'].unsqueeze(0).to(device)
    garment = sample_data['garment'].unsqueeze(0).to(device)
    original_garment_person = sample_data['original_garment_person'].unsqueeze(0).to(device)
    
    print("Input shapes:")
    print(f"  Target image: {target_image.shape}")
    print(f"  Cloth agnostic: {cloth_agnostic.shape}")
    print(f"  Garment: {garment.shape}")
    print(f"  Original garment person: {original_garment_person.shape}")
    
    # Generate synthesized image
    print("Running inference...")
    synthesized = sample_with_conditions_cfg(
        diffusion, model, cloth_agnostic, garment, 
        conf.img_size, guidance_scale=guidance_scale, 
        steps=sampling_steps, seed=seed
    )
    
    print(f"Generated image shape: {synthesized.shape}")
    
    # Save individual images
    target_path = os.path.join(output_dir, f'{sample_id}_original.png')
    cloth_agnostic_path = os.path.join(output_dir, f'{sample_id}_cloth_agnostic.png')
    garment_path = os.path.join(output_dir, f'{sample_id}_garment.png')
    original_garment_person_path = os.path.join(output_dir, f'{sample_id}_original_garment_person.png')
    synthesized_path = os.path.join(output_dir, f'{sample_id}_synthesized.png')
    
    save_image_from_tensor(target_image.squeeze(0), target_path)
    save_image_from_tensor(cloth_agnostic[:, :3].squeeze(0), cloth_agnostic_path)  # Only RGB
    save_image_from_tensor(garment[:, :3].squeeze(0), garment_path)  # Only RGB
    save_image_from_tensor(original_garment_person.squeeze(0), original_garment_person_path)
    save_image_from_tensor(synthesized.squeeze(0), synthesized_path)
    
    print(f"Saved images:")
    print(f"  Original: {target_path}")
    print(f"  Cloth agnostic: {cloth_agnostic_path}")
    print(f"  Garment: {garment_path}")
    print(f"  Original garment person: {original_garment_person_path}")
    print(f"  Synthesized: {synthesized_path}")
    
    # Create comparison grid
    grid_path = os.path.join(output_dir, f'{sample_id}_comparison.png')
    create_comparison_grid(
        target_image.squeeze(0), 
        cloth_agnostic.squeeze(0), 
        garment.squeeze(0), 
        original_garment_person.squeeze(0), 
        synthesized.squeeze(0), 
        grid_path
    )
    
    print(f"Inference completed successfully!")
    return synthesized


def main():
    parser = argparse.ArgumentParser(description='Run inference on DiT Virtual Try-on model')
    parser.add_argument('--model_dir', type=str, default='model_4', 
                        help='Directory containing the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save inference results')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample to process')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Scale for classifier-free guidance')
    parser.add_argument('--sampling_steps', type=int, default=100,
                        help='Number of sampling steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible generation')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    run_inference(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_idx=args.sample_idx,
        guidance_scale=args.guidance_scale,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main() 