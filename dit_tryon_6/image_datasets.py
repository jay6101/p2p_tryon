from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
import PIL
import os
import numpy as np
import torch
import json
import cv2
import random
from PIL import Image
    
def _list_image_files(data_dir):
    results = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            ext = file.split(".")[-1]
            if "." in file and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                # Construct the full path of the file
                full_path = os.path.join(root, file)
                results.append(full_path)
    return results

def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class ImageLoader(Dataset):
    def __init__(self, image_paths, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform
      
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class TryonDataset(Dataset):
    def __init__(self, data_dir, img_size, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Base transform without augmentation
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])
        
        # Transform for garment only (no augmentation)
        self.garment_transform = self.base_transform
        
        # Custom transform will be applied at runtime to ensure
        # the same random augmentation is applied to related images
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Rotation range in degrees
        self.max_rotation = 10
        
    def _load_metadata(self):
        # This method would normally load a JSON file with dataset annotations
        metadata = {}
        sample_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for sample_id in sample_dirs:
            sample_dir = os.path.join(self.data_dir, sample_id)
            metadata[sample_id] = {
                'target_path': os.path.join(sample_dir, '1.jpg'),
                'cloth_agnostic_mask_path': os.path.join(sample_dir, '1/alpha/1_new.png'),
                'garment_path': os.path.join(sample_dir, '2.jpg'),
                'garment_mask_path': os.path.join(sample_dir, '2/alpha/1.png'),
                'garment_pose': os.path.join(sample_dir, '2_pose.jpg'),
                'target_pose': os.path.join(sample_dir, '1_pose.jpg')
            }
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)

    def apply_augmentation(self, imgs):
        """Apply the same random augmentation to a list of images"""
        # Determine random augmentation parameters
        do_flip = random.random() > 0.5
        rotation_angle = random.uniform(-self.max_rotation, self.max_rotation)
        
        # Convert numpy images to PIL for transformations
        pil_imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
        
        # Apply the same transformations to all images
        augmented_imgs = []
        for img in pil_imgs:
            # Apply horizontal flip
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Apply rotation
            if rotation_angle != 0:
                img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=False)
            
            # Convert back to numpy
            augmented_imgs.append(np.array(img) / 255.0)
        
        return augmented_imgs
    
    def __getitem__(self, idx):
        sample_id = list(self.metadata.keys())[idx]
        sample_data = self.metadata[sample_id]
        
        # Load images
        target_img = np.array(PIL.Image.open(sample_data['target_path']))/255.0
        garment_img = np.array(PIL.Image.open(sample_data['garment_path']))/255.0
        
        try:
            cloth_agnostic_mask = np.array(PIL.Image.open(sample_data['cloth_agnostic_mask_path']))/255.0
        except:
            try:
                cloth_agnostic_mask = np.array(PIL.Image.open(sample_data['cloth_agnostic_mask_path'].replace('1.png', '3.png')))/255.0
            except:
                try:
                    cloth_agnostic_mask = np.array(PIL.Image.open(sample_data['cloth_agnostic_mask_path'].replace('1.png', '2.png')))/255.0
                except:
                    cloth_agnostic_mask = np.zeros((1024,768))
        cloth_agnostic_mask = 1.0 - cloth_agnostic_mask
        cloth_agnostic_img = (cloth_agnostic_mask[:,:,None])*(target_img)
        
        try:
            garment_mask = np.array(PIL.Image.open(sample_data['garment_mask_path']))/255.0
        except:
            try:
                garment_mask = np.array(PIL.Image.open(sample_data['garment_mask_path'].replace('1.png', '3.png')))/255.0
            except:
                try:
                    garment_mask = np.array(PIL.Image.open(sample_data['garment_mask_path'].replace('1.png', '2.png')))/255.0
                except:
                    garment_mask = np.zeros((1024,768))
        garment_img = (garment_mask[:,:,None])*(garment_img)
        
        # Load pose data
        target_pose = np.array(PIL.Image.open(sample_data['target_pose']))/255.0
        garment_pose = np.array(PIL.Image.open(sample_data['garment_pose']))/255.0
        
        # Apply the same augmentation to target image, cloth agnostic, and target pose
        target_img, cloth_agnostic_img, target_pose, garment_img, garment_pose = self.apply_augmentation([
            target_img, cloth_agnostic_img, target_pose, garment_img, garment_pose
        ])
        
        # Apply transforms
        target_tensor = self.base_transform(target_img.astype(np.float32))
        cloth_agnostic_tensor = self.base_transform(cloth_agnostic_img.astype(np.float32))
        target_pose_tensor = self.base_transform(target_pose.astype(np.float32))
        
        # Apply non-augmented transform to garment and garment pose
        garment_tensor = self.garment_transform(garment_img.astype(np.float32))
        garment_pose_tensor = self.garment_transform(garment_pose.astype(np.float32))
        
        cloth_agnostic_tensor = torch.cat([cloth_agnostic_tensor, target_pose_tensor], dim=0)
        garment_tensor = torch.cat([garment_tensor, garment_pose_tensor], dim=0)
        
        return {
            'target_image': target_tensor,
            'cloth_agnostic': cloth_agnostic_tensor,
            'garment': garment_tensor,
            'sample_id': sample_id
        }
        
def create_loader(data_dir, img_size, batch_size):
    all_files = _list_image_files(data_dir)
    transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = ImageLoader(all_files, transform)
    loader = iter(DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=4, pin_memory=True
    ))
    return loader

def create_tryon_loader(data_dir, img_size, batch_size):
    dataset = TryonDataset(data_dir, img_size)
    loader = iter(DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=4, pin_memory=True
    ))
    return loader
