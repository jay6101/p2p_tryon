'''
Simple example script to run inference on a few samples
This shows how to use the inference.py script programmatically
'''
import os
import torch
from inference import sample_with_conditions_cfg, save_image_from_tensor, create_comparison_grid, tensor_to_pil
from image_data_infer import TryonDataset
from config import config
from dit_4 import DiT
from utils import Config
from diff_utils import Diffusion
import random

class InferenceModel:
    """
    Wrapper class to load model once and reuse for multiple inferences
    """
    def __init__(self, model_dir, device='cuda:1'):
        # Load config
        self.conf = Config(config, model_dir)
        self.device = device
        
        # Initialize model
        self.model = DiT(
            img_size=self.conf.img_size, 
            dim=self.conf.dim, 
            patch_size=self.conf.patch_size,
            depth=self.conf.depth, 
            heads=self.conf.heads, 
            mlp_dim=self.conf.mlp_dim, 
            k=self.conf.k
        )
        
        # Load checkpoint
        ckpt_path = os.path.join(model_dir, 'last_ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Load EMA model weights (EMA typically gives better results for inference)
        if 'ema' in ckpt:
            self.model.load_state_dict(ckpt['ema'])
            print("Loaded EMA model weights")
        else:
            self.model.load_state_dict(ckpt['model'])
            print("Loaded regular model weights")
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize diffusion
        self.diffusion = Diffusion()
        
        print(f"Model loaded successfully on {device}")
    
    def run_inference_single(self, data_dir, output_dir, sample_idx=0, guidance_scale=3.0, 
                           sampling_steps=100, seed=42):
        """
        Run inference on a single sample using the pre-loaded model
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        dataset = TryonDataset(data_dir, self.conf.img_size)
        
        if sample_idx >= len(dataset):
            raise ValueError(f"Sample index {sample_idx} exceeds dataset size {len(dataset)}")
        
        # Get a sample
        sample_data = dataset[sample_idx]
        sample_id = sample_data['target_sample_id']
        
        print(f"Processing sample: {sample_id}")
        
        # Prepare data
        target_image = sample_data['target_image'].unsqueeze(0).to(self.device)  # Add batch dimension
        cloth_agnostic = sample_data['cloth_agnostic'].unsqueeze(0).to(self.device)
        garment = sample_data['garment'].unsqueeze(0).to(self.device)
        original_garment_person = sample_data['original_garment_person'].unsqueeze(0).to(self.device)
        
        print("Input shapes:")
        print(f"  Target image: {target_image.shape}")
        print(f"  Cloth agnostic: {cloth_agnostic.shape}")
        print(f"  Garment: {garment.shape}")
        print(f"  Original garment person: {original_garment_person.shape}")
        
        # Generate synthesized image
        print("Running inference...")
        synthesized = sample_with_conditions_cfg(
            self.diffusion, self.model, cloth_agnostic, garment, 
            self.conf.img_size, guidance_scale=guidance_scale, 
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
    # Configuration
    model_dir = 'model_4'
    data_dir = '/space/mcdonald-syn01/1/projects/jsawant/ECE285/data/custom_train_data'  # UPDATE THIS PATH
    output_dir = 'inference_results_fid'
    
    # Make sure CUDA is available
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model once
    print("Loading model...")
    inference_model = InferenceModel(model_dir, device)
    print("Model loaded! Starting inference on samples...")
    
    # Run inference on first few samples
    random_sample_idx = random.sample(range(20001), 2000-350) 
    for sample_idx in random_sample_idx:  # Process first 3 samples (or fewer if dataset is small)
        try:
            print(f"\n{'='*50}")
            print(f"Processing sample {sample_idx}")
            print(f"{'='*50}")
            
            result = inference_model.run_inference_single(
                data_dir=data_dir,
                output_dir=f"{output_dir}/sample_{sample_idx:03d}",
                sample_idx=sample_idx,
                guidance_scale=3.0,  # Higher values = stronger guidance
                sampling_steps=128,   # Reduced for faster inference (100 is better quality)
                seed=42 + sample_idx,  # Different seed for each sample
            )
            
            print(f"Successfully processed sample {sample_idx}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"\nInference completed! Results saved in: {output_dir}")
    print(f"Each sample folder contains:")
    print(f"  - *_original.png: Original target image")
    print(f"  - *_cloth_agnostic.png: Person without clothes")
    print(f"  - *_garment.png: Garment to try on")
    print(f"  - *_synthesized.png: Generated try-on result")
    print(f"  - *_comparison.png: Side-by-side comparison grid")

if __name__ == '__main__':
    main() 