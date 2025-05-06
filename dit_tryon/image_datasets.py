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
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size[0], img_size[1])),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
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
                'garment_pose_json': os.path.join(sample_dir, '2.json'),
                'target_pose_json': os.path.join(sample_dir, '1.json')
            }
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)

    
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
        # Apply transforms
        target_tensor = self.transform(target_img.astype(np.float32))
        cloth_agnostic_tensor = self.transform(cloth_agnostic_img.astype(np.float32))
        garment_tensor = self.transform(garment_img.astype(np.float32))
        
        # Load pose data from meta.json
        with open(sample_data['target_pose_json'], 'r') as f:
            target_pose = json.load(f)
            
        with open(sample_data['garment_pose_json'], 'r') as f:
            garment_pose = json.load(f)
            
        target_pose = np.array(target_pose['people'][0]['pose_keypoints_2d'], dtype=np.float32)
        garment_pose = np.array(garment_pose['people'][0]['pose_keypoints_2d'], dtype=np.float32)
        
        target_pose[0::3] /= 1024
        target_pose[1::3] /= 768
        target_pose = torch.from_numpy(target_pose).float()
        
        garment_pose[0::3] /= 1024
        garment_pose[1::3] /= 768
        garment_pose = torch.from_numpy(garment_pose).float()   
        
        # Ensure pose vectors are the correct size (150)
        if garment_pose.size(0) < 150:
            garment_pose = torch.cat([garment_pose, torch.zeros(150 - garment_pose.size(0))], dim=0)
        elif garment_pose.size(0) > 150:
            garment_pose = garment_pose[:150]
            
        if target_pose.size(0) < 150:
            target_pose = torch.cat([target_pose, torch.zeros(150 - target_pose.size(0))], dim=0)
        elif target_pose.size(0) > 150:
            target_pose = target_pose[:150]
        
        return {
            'target_image': target_tensor,
            'cloth_agnostic': cloth_agnostic_tensor,
            'garment': garment_tensor,
            'garment_pose': garment_pose,
            'target_pose': target_pose,
            'sample_id': sample_id
        }
        
    @staticmethod
    def expand_and_clean_mask(mask, radius=30):
        """
        Expand the 1s region by `radius` pixels (dilation), then keep only the largest connected component.
        """

        # 2. Dilation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # 3. Optional smoothing + rethreshold (for clean edges)
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (7, 7), sigmaX=2)
        binary = (blurred > 0.5).astype(np.uint8)

        # 4. Connected‑components analysis
        #    `num_labels` includes the background label 0
        num_labels, labels = cv2.connectedComponents(binary)

        if num_labels <= 1:
            # no foreground at all
            return binary

        # 5. Find the largest component (exclude background label 0)
        #    Compute area of each label id
        areas = [
            np.sum(labels == lbl)
            for lbl in range(1, num_labels)
        ]
        largest_lbl = 1 + int(np.argmax(areas))

        # 6. Build a cleaned mask that keeps only that label
        cleaned = (labels == largest_lbl).astype(np.uint8)

        return cleaned

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
