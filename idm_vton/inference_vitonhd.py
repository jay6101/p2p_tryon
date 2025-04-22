import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from tqdm import tqdm
import json
import shutil

# Import components from Change-Clothes-AI
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
from torchvision.transforms.functional import to_pil_image
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Define a VitonHD dataset class
class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path,
        paired=True,
        size=(1024, 768),
        transform=None,
        test_pairs_file=None
    ):
        self.dataroot = dataroot_path
        self.paired = paired
        self.height, self.width = size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.toTensor = transforms.ToTensor()
        
        # Load tagged data if available
        self.annotations = {}
        json_path = os.path.join(dataroot_path, "train/vitonhd_train_tagged.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                data = json.load(file)
                
            annotation_list = ["sleeveLength", "neckLine", "item"]
            
            for k, v in data.items():
                for elem in v:
                    annotation_str = ""
                    for template in annotation_list:
                        for tag in elem["tag_info"]:
                            if tag["tag_name"] == template and tag["tag_category"] is not None:
                                annotation_str += tag["tag_category"]
                                annotation_str += " "
                    self.annotations[elem["file_name"]] = annotation_str
        
        # Load pairs data
        self.im_names = []
        self.c_names = []
        
        if test_pairs_file:
            pairs_path = test_pairs_file
        else:
            pairs_path = os.path.join(dataroot_path, "train_pairs.txt")
            
        with open(pairs_path, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                if paired:
                    # For paired testing, use the same image name for both
                    self.im_names.append(im_name)
                    self.c_names.append(im_name)
                else:
                    # For unpaired testing, use as specified in pairs file
                    self.im_names.append(im_name)
                    self.c_names.append(c_name)
                    
        self.clip_processor = CLIPImageProcessor()

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        # Get cloth description if available, or use default
        if c_name in self.annotations:
            cloth_annotation = self.annotations[c_name]
        else:
            cloth_annotation = "clothing item"
            
        # Load cloth image
        cloth_path = os.path.join(self.dataroot, "train", "cloth", c_name)
        cloth = Image.open(cloth_path).convert("RGB")
        
        # Load person image
        image_path = os.path.join(self.dataroot, "train", "image", im_name)
        human_img = Image.open(image_path).convert("RGB").resize((self.width, self.height))
        
        # Load cloth mask if available
        cloth_mask_path = os.path.join(self.dataroot, "train", "cloth-mask", c_name)
        if os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert("L")
        else:
            cloth_mask = None
            
        return {
            "cloth": cloth,
            "cloth_name": c_name,
            "image": human_img,
            "image_name": im_name,
            "cloth_annotation": cloth_annotation,
            "cloth_mask": cloth_mask
        }

    def __len__(self):
        return len(self.im_names)

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Change-Clothes-AI using VITONHD dataset")
    parser.add_argument("--vitonhd_path", type=str, required=True, help="Path to VITONHD dataset")
    parser.add_argument("--output_dir", type=str, default="vitonhd_results", help="Directory to save results")
    parser.add_argument("--test_pairs_file", type=str, default=None, help="Custom test pairs file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None for all)")
    parser.add_argument("--paired", action="store_true", help="Use paired images (default: unpaired)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--denoise_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--use_fp32", action="store_true", help="Use FP32 precision instead of FP16")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # For fp16 consistency, we'll force fp16 even if use_fp32 is set
    dtype = torch.float16
    print(f"Using precision: {dtype}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    dataset = VitonHDDataset(
        dataroot_path=args.vitonhd_path,
        paired=args.paired,
        test_pairs_file=args.test_pairs_file
    )
    
    if args.num_samples is not None:
        # Limit the number of samples
        indices = list(range(min(args.num_samples, len(dataset))))
    else:
        # Use all samples
        indices = list(range(len(dataset)))
        #indices = indices[:4941]
        
    print(f"Processing {len(indices)} samples from VITONHD test dataset")
    
    # Load models
    # Base model path
    base_path = 'yisol/IDM-VTON'
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Load UNet models
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=dtype,
    )
    unet.requires_grad_(False)
    
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=dtype,
    )
    UNet_Encoder.requires_grad_(False)
    
    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    
    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder_one.requires_grad_(False)
    
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    )
    text_encoder_two.requires_grad_(False)
    
    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=dtype,
    )
    image_encoder.requires_grad_(False)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.requires_grad_(False)
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    # Load preprocessing models
    parsing_model = Parsing(args.gpu_id)
    openpose_model = OpenPose(args.gpu_id)
    
    # Create pipeline
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    pipe.unet_encoder = UNet_Encoder
    
    # Move models to device
    pipe.to(device)
    pipe.unet_encoder.to(device)
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    # Process samples
    for z,idx in tqdm(enumerate(indices)):
        if z<=4940:
            continue
        sample = dataset[idx]
        human_img = sample["image"]
        garm_img = sample["cloth"]
        garment_des = sample["cloth_annotation"]
        im_name = sample["image_name"]
        c_name = sample["cloth_name"]
        
        # Resize images
        human_img = human_img.resize((768, 1024))
        garm_img = garm_img.convert("RGB").resize((768, 1024))
        
        try:
            # Generate mask using automatic detection
            # Detect keypoints
            keypoints = openpose_model(human_img.resize((384, 512)))
            # Parse human body parts
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            
            # Determine garment category based on description
            if "dress" in garment_des.lower():
                category = "dresses"
            elif "pant" in garment_des.lower() or "trouser" in garment_des.lower() or "jean" in garment_des.lower():
                category = "lower_body"
            else:
                category = "upper_body"
                
            # Generate mask based on category
            mask, _ = get_mask_location('hd', category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
            
            # Convert mask to PIL first
            mask_pil = mask
            
            # Generate grayscale mask for visualization (on CPU to avoid dtype issues)
            mask_gray = Image.new("RGB", mask_pil.size)
            mask_gray.paste(mask_pil, (0, 0))
            
            # Generate prompt
            prompt = f"((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), the model is wearing {garment_des}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch"
            
            # Process input images
            pose_img = generate_pose_image(human_img, device)
            
            # Convert inputs to tensors of the appropriate type
            pose_img_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
            mask_tensor = tensor_transform(mask_pil).unsqueeze(0).to(device, dtype)
            human_tensor = tensor_transform(human_img).unsqueeze(0).to(device, dtype)
            
            # Encode prompt without autocast to maintain precision
            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                # Generate cloth-specific prompt
                cloth_prompt = f"((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), a photo of {garment_des}"
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    cloth_prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )
                
                # Run inference with tensors that all have the same dtype
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                    num_inference_steps=args.denoise_steps,
                    generator=torch.Generator(device).manual_seed(args.seed),
                    strength=1.0,
                    pose_img=pose_img_tensor,
                    text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                    cloth=garm_tensor,
                    mask_image=mask_tensor,
                    image=human_tensor,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img,  # This one can stay as PIL
                    guidance_scale=args.guidance_scale,
                )[0]
            
            # Save results
            os.mkdir(os.path.join(args.output_dir,f"{z}"))
            #output_filename = f"{os.path.splitext(im_name)[0]}_{os.path.splitext(c_name)[0]}.jpg"
            output_filename = f"2.jpg"
            og_img = f"/space/mcdonald-syn01/1/projects/jsawant/ECE285/VITONHD/train/image/{c_name}"
            shutil.copy(og_img, os.path.join(args.output_dir,f"{z}/1.jpg"))
            #mask_filename = f"{os.path.splitext(im_name)[0]}_{os.path.splitext(c_name)[0]}_mask.jpg"
            
            # Save the result image
            result_path = os.path.join(args.output_dir, f"{z}", output_filename)
            images[0].save(result_path)
            
            # Save the mask image
            # mask_path = os.path.join(args.output_dir, mask_filename)
            # mask_gray.save(mask_path)
            
            print(f"Saved result to {result_path}")
            
        except Exception as e:
            print(f"Error processing {im_name} with {c_name}: {e}")
            continue

def generate_pose_image(human_img, device):
    """Generate the pose image using DensePose"""
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    
    # Prepare the image
    human_img_resized = human_img.resize((384, 512))
    human_img_arg = _apply_exif_orientation(human_img_resized)
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    # Use DensePose to generate pose information
    args = apply_net.create_argument_parser().parse_args((
        'show',
        './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl',
        'dp_segm',
        '-v',
        '--opts',
        'MODEL.DEVICE',
        device
    ))
    
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]  # BGR to RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    return pose_img

if __name__ == "__main__":
    main() 