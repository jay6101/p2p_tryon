from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from tqdm import tqdm

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

data_path = '/space/mcdonald-syn01/1/projects/jsawant/ECE285/dit_tryon_2/data/custom_train_data/'
for folder in tqdm(os.listdir(data_path)):
    if os.path.exists(os.path.join(data_path, folder, '1/alpha/1_new.png')):
        continue
    try:
        mask_path = os.path.join(data_path, folder, '1/alpha/1.png')
        mask = np.array(Image.open(mask_path))/255.0
    except:
        try:
            mask_path = os.path.join(data_path, folder, '1/alpha/3.png')
            mask = np.array(Image.open(mask_path))/255.0
        except:
            mask_path = os.path.join(data_path, folder, '1/alpha/2.png')
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))/255.0
            else:
                continue
    
    mask = expand_and_clean_mask(mask)
    mask = Image.fromarray(mask*255).convert('L')
    save_path = os.path.join(data_path, folder, "1/alpha/1_new.png")
    mask.save(save_path)
    
    if os.path.exists(os.path.join(data_path, folder, '2/alpha/1_new.png')):
        continue
    try:
        mask_path = os.path.join(data_path, folder, '2/alpha/1.png')
        mask = np.array(Image.open(mask_path))/255.0
    except:
        try:
            mask_path = os.path.join(data_path, folder, '2/alpha/3.png')
            mask = np.array(Image.open(mask_path))/255.0
        except:
            mask_path = os.path.join(data_path, folder, '2/alpha/2.png')
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))/255.0
            else:
                continue
    mask = expand_and_clean_mask(mask)
    mask = Image.fromarray(mask*255).convert('L')
    save_path = os.path.join(data_path, folder, "2/alpha/1_new.png")
    mask.save(save_path)

