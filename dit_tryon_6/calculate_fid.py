#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import glob
from typing import Union

# Ignite imports for metrics
from ignite.metrics import FID
import ignite.distributed as idist

import torch
from packaging.version import Version

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class InceptionModel(torch.nn.Module):
    r"""Inception Model pre-trained on the ImageNet Dataset.

    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def __init__(self, return_features: bool, device: Union[str, torch.device] = "cpu") -> None:
        try:
            import torchvision
            from torchvision import models
        except ImportError:
            raise ModuleNotFoundError("This module requires torchvision to be installed.")
        super(InceptionModel, self).__init__()
        self._device = device
        if Version(torchvision.__version__) < Version("0.13.0"):
            model_kwargs = {"pretrained": True}
        else:
            model_kwargs = {"weights": models.Inception_V3_Weights.DEFAULT}

        self.model = models.inception_v3(**model_kwargs).to(self._device)

        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        if data.device != torch.device(self._device):
            data = data.to(self._device)
        return self.model(data)

def count_synthesized_images(inference_results_dir):
    """Count the number of synthesized images available"""
    pattern = os.path.join(inference_results_dir, "sample_*", "*_synthesized.png")
    image_paths = glob.glob(pattern)
    return len(image_paths)

def load_synthesized_images(inference_results_dir, transform=None, num_samples=None):
    """Load synthesized images from inference_results_fid/sample_XXX/YYY_synthesized.png"""
    print(f"Loading synthesized images from: {inference_results_dir}")
    
    # Check if the directory exists
    if not os.path.exists(inference_results_dir):
        raise FileNotFoundError(f"Inference results directory {inference_results_dir} does not exist")
    
    # Get all synthesized image files
    pattern = os.path.join(inference_results_dir, "sample_*", "*_synthesized.png")
    image_paths = glob.glob(pattern)
    image_paths.sort()  # Sort to ensure consistent ordering
    
    if len(image_paths) == 0:
        raise ValueError(f"No synthesized images found in {inference_results_dir}")
    
    print(f"Found {len(image_paths)} synthesized images")
    
    if num_samples is not None and num_samples < len(image_paths):
        image_paths = image_paths[:num_samples]
        print(f"Using first {num_samples} synthesized images")
    
    # Load images
    synthesized_images = []
    
    print(f"Loading synthesized images...")
    for i, img_path in enumerate(image_paths):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transform if provided
            if transform:
                image = transform(image)
            else:
                # Default transform: convert to tensor and normalize to [0, 1]
                image = transforms.ToTensor()(image)
            
            synthesized_images.append(image)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                print(f"Loaded {i + 1}/{len(image_paths)} synthesized images...")
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    if len(synthesized_images) == 0:
        raise ValueError("No synthesized images could be loaded successfully")
    
    # Convert to tensor
    synthesized_images = torch.stack(synthesized_images)
    
    # Ensure images are in [0, 1] range
    synthesized_images = torch.clamp(synthesized_images, 0, 1)
    
    print(f"Successfully loaded {len(synthesized_images)} synthesized images")
    print(f"Image shape: {synthesized_images.shape}")
    print(f"Image range: [{synthesized_images.min().item():.3f}, {synthesized_images.max().item():.3f}]")
    
    return synthesized_images

def load_real_images(real_data_dir, transform=None, num_samples=None):
    """Load real images from custom_train_data/*/1.jpg"""
    print(f"Loading real images from: {real_data_dir}")
    
    # Check if the directory exists
    if not os.path.exists(real_data_dir):
        raise FileNotFoundError(f"Real data directory {real_data_dir} does not exist")
    
    # Get all real image files
    pattern = os.path.join(real_data_dir, "*", "1.jpg")
    image_paths = glob.glob(pattern)
    image_paths.sort()  # Sort to ensure consistent ordering
    
    if len(image_paths) == 0:
        raise ValueError(f"No real images found in {real_data_dir}")
    
    print(f"Found {len(image_paths)} real images")
    
    if num_samples is not None and num_samples < len(image_paths):
        image_paths = image_paths[:num_samples]
        print(f"Using first {num_samples} real images")
    
    # Load images
    real_images = []
    
    print(f"Loading real images...")
    for i, img_path in enumerate(image_paths):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transform if provided
            if transform:
                image = transform(image)
            else:
                # Default transform: convert to tensor and normalize to [0, 1]
                image = transforms.ToTensor()(image)
            
            real_images.append(image)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                print(f"Loaded {i + 1}/{len(image_paths)} real images...")
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    
    if len(real_images) == 0:
        raise ValueError("No real images could be loaded successfully")
    
    # Convert to tensor
    real_images = torch.stack(real_images)
    
    # Ensure images are in [0, 1] range
    real_images = torch.clamp(real_images, 0, 1)
    
    print(f"Successfully loaded {len(real_images)} real images")
    print(f"Image shape: {real_images.shape}")
    print(f"Image range: [{real_images.min().item():.3f}, {real_images.max().item():.3f}]")
    
    return real_images

def interpolate(batch):
    """Resize images to 299x299 for Inception model"""
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img.cpu())
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        tensor_img = transforms.ToTensor()(resized_img)
        # Normalize for Inception model (expects [-1, 1])
        normalized_img = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )(tensor_img)
        arr.append(normalized_img)
    return torch.stack(arr).to(device)

def calculate_fid(real_images, synthesized_images, batch_size=64):
    """Calculate FID score with batch processing for memory efficiency"""
    print(f"Calculating FID with {len(real_images)} real images and {len(synthesized_images)} synthesized images...")
    
    # Initialize FID metric
    print("Initializing FID metric...")
    feat_extractor = InceptionModel(return_features=True, device=device)
    fid_metric = FID(num_features=2048, feature_extractor=feat_extractor, device=device)
    fid_metric.reset()
    
    # Use equal number of images for both sets
    num_images = min(len(real_images), len(synthesized_images))
    num_batches = (num_images + batch_size - 1) // batch_size
    
    print(f"Processing {num_images} images in {num_batches} batches of size {batch_size}...")
    
    for i in range(0, num_images, batch_size):
        end_idx = min(i + batch_size, num_images)
        batch_size_actual = end_idx - i
        
        print(f"Processing batch {i // batch_size + 1}/{num_batches}, images {i}-{end_idx-1}...")
        
        # Get batch of images
        real_batch = real_images[i:end_idx]
        synth_batch = synthesized_images[i:end_idx]
        
        # Resize images for Inception model
        print("  Resizing images for Inception model...")
        real_batch_resized = interpolate(real_batch)
        synth_batch_resized = interpolate(synth_batch)
        
        # Update FID metric with batch
        print("  Updating FID metric...")
        fid_metric.update((real_batch_resized, synth_batch_resized))
        
        # Free up memory
        del real_batch_resized, synth_batch_resized
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute final FID score
    print("Computing final FID score...")
    fid_score = fid_metric.compute()
    
    return fid_score

def main():
    parser = argparse.ArgumentParser(description="Calculate FID between synthesized and real images")
    parser.add_argument("--inference_results_dir", type=str, 
                        default="/space/mcdonald-syn01/1/projects/jsawant/ECE285/dit_tryon_6/inference_results_fid",
                        help="Path to inference results directory containing synthesized images")
    parser.add_argument("--real_data_dir", type=str, 
                        default="/space/mcdonald-syn01/1/projects/jsawant/ECE285/data/custom_train_data",
                        help="Path to real data directory")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to use for evaluation (use all if not specified)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for resizing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # First, determine how many samples to use
    print("Determining number of samples to use...")
    num_synth_available = count_synthesized_images(args.inference_results_dir)
    print(f"Found {num_synth_available} synthesized images available")
    
    # Determine the actual number of samples to use
    if args.num_samples is not None:
        num_samples_to_use = min(args.num_samples, num_synth_available)
    else:
        num_samples_to_use = num_synth_available
    
    print(f"Will use {num_samples_to_use} images for both synthesized and real datasets")
    
    # Initialize data transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    # Load equal numbers of synthesized and real images
    print("Loading synthesized images...")
    synthesized_images = load_synthesized_images(
        args.inference_results_dir, 
        transform=transform,
        num_samples=num_samples_to_use
    )
    
    print("Loading real images...")
    real_images = load_real_images(
        args.real_data_dir, 
        transform=transform,
        num_samples=num_samples_to_use  # Load exactly the same number as synthesized
    )
    
    # Verify we have equal numbers
    print(f"Loaded {len(synthesized_images)} synthesized images and {len(real_images)} real images")
    
    # Calculate FID score
    print("Calculating FID...")
    fid_score = calculate_fid(real_images, synthesized_images, batch_size=args.batch_size)
    
    print(f"FID Score: {fid_score:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "fid_evaluation_results.txt")
    print(f"Saving results to {results_path}")
    with open(results_path, "w") as f:
        f.write(f"FID Evaluation Results\n")
        f.write(f"======================\n")
        f.write(f"Inference results dir: {args.inference_results_dir}\n")
        f.write(f"Real data dir: {args.real_data_dir}\n")
        f.write(f"Number of synthesized images: {len(synthesized_images)}\n")
        f.write(f"Number of real images: {len(real_images)}\n")
        f.write(f"Image size: {args.img_size}x{args.img_size}\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
    
    # Save results as JSON for easy parsing
    results_json_path = os.path.join(args.output_dir, "fid_evaluation_results.json")
    results_dict = {
        "inference_results_dir": args.inference_results_dir,
        "real_data_dir": args.real_data_dir,
        "num_synthesized_images": len(synthesized_images),
        "num_real_images": len(real_images),
        "img_size": args.img_size,
        "fid_score": float(fid_score)
    }
    
    with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results also saved to {results_json_path}")
    
    # Visualize some samples for comparison
    n_samples = min(8, len(synthesized_images), len(real_images))
    plt.figure(figsize=(16, 8))

    # Top row: real images
    plt.subplot(2, 1, 1)
    plt.title("Real Images")
    grid_real = torchvision.utils.make_grid(real_images[:n_samples], nrow=n_samples)
    plt.imshow(grid_real.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

    # Bottom row: synthesized images
    plt.subplot(2, 1, 2)
    plt.title("Synthesized Images")
    grid_synth = torchvision.utils.make_grid(synthesized_images[:n_samples], nrow=n_samples)
    plt.imshow(grid_synth.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

    comparison_path = os.path.join(args.output_dir, "fid_sample_comparison.png")
    print(f"Saving comparison image to {comparison_path}")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("FID evaluation completed successfully!")

if __name__ == "__main__":
    main() 