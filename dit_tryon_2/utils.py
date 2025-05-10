import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def deprocess(img):
    return img * 127.5 + 127.5

def plot_batch(tensor, batch_data, plot_shape, filename, img_size=(32,32)):
    tensor_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
    tensor_np = np.clip(tensor_np, 0, 255).astype(np.uint8)
    rows = plot_shape[0]
    #columns = plot_shape[1]
    columns = 4
    # Proportional scale by img size
    scale = (img_size[0] / 90) 
    fig, axes = plt.subplots(rows, columns, figsize=(columns * scale, rows * scale))
    
    # Iterate through each cell in the grid and plot images
    for i in range(rows):
        for j in range(columns):
            # Calculate the index of the image
            img_idx = i
            target_img = batch_data['target_image'][img_idx]
            garment_img = batch_data['garment'][img_idx]
            cloth_agnostic_mask = batch_data['cloth_agnostic'][img_idx]
            # If the image index exceeds the number of images, stop plotting
            if img_idx >= tensor_np.shape[0]:
                break
            
            # Display the image in the respective subplot
            ax = axes[i, j]
            if j == 0:
                target_img = deprocess(target_img[:3,:,:].permute(1, 2, 0))
                target_img = np.clip(target_img.numpy(), 0, 255).astype(np.uint8)
                ax.imshow(target_img)
            elif j == 1:
                cloth_agnostic_mask = deprocess(cloth_agnostic_mask[:3,:,:].permute(1, 2, 0))
                cloth_agnostic_mask = np.clip(cloth_agnostic_mask.numpy(), 0, 255).astype(np.uint8)
                ax.imshow(cloth_agnostic_mask)
            elif j == 2:
                garment_img = deprocess(garment_img[:3,:,:].permute(1, 2, 0))
                garment_img = np.clip(garment_img.numpy(), 0, 255).astype(np.uint8)
                ax.imshow(garment_img)
            else:
                ax.imshow(tensor_np[img_idx])
            ax.axis('off')
    
    # Adjust the layout to minimize whitespace
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(filename)
    plt.close()

class Config(object):
    def __init__(self, input_dict, save_dir):
        file_path = os.path.join(save_dir, "config.json")
        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            for key, value in input_dict.items():
                setattr(self, key, value)
            self.save_config(file_path, save_dir)
        self.print_variables()

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')

    def print_variables(self):
        # Print all variables (attributes) of the Config object
        for key, value in vars(self).items():
            print(f"{key}: {value}")
