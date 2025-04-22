import spaces
import gradio as gr
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List


import torch
import os
from transformers import AutoTokenizer

import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
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
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

@spaces.GPU
def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed, category):
    """虚拟试衣主函数
    Args:
        dict: 输入图像字典，包含背景和图层信息
        garm_img: 服装图片
        garment_des: 服装描述文本
        is_checked: 是否启用自动检测模式
        is_checked_crop: 是否启用图像裁剪
        denoise_steps: 去噪步数
        seed: 随机种子
        category: 服装类别
    Returns:
        生成的试衣结果图像和灰度遮罩
    """
    # 1. 初始化和设备设置 - 使用GPU进行处理
    device = "cuda"
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # 2. 图像预处理 - 调整服装和人物图像大小
    garm_img = garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")
    orig_size = human_img_orig.size  # 保存原始尺寸

    # 2.1 如果启用裁剪，按3:4比例裁剪人物图像
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    # 3. 生成遮罩
    if is_checked:
        # 3.1 使用自动检测模式
        # 使用OpenPose检测人体关键点
        keypoints = openpose_model(human_img.resize((384,512)))
        # 使用解析模型生成人体部位解析
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        # 根据类别和关键点生成遮罩
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        # 3.2 使用手动提供的遮罩
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
    
    # 3.3 生成灰度遮罩
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    # 4. 姿态处理
    # 4.1 调整图像方向并转换格式
    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # 4.2 使用DensePose生成姿态信息
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))

    # 5. AI生成过程
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 5.1 生成正面提示词嵌入
                prompt = "((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), the model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch"
                with torch.inference_mode():
                    # 编码提示词
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    # 5.2 生成服装相关的提示词嵌入
                    prompt = "((best quality, masterpiece, ultra-detailed, high quality photography, photo realistic)), a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    # 5.3 准备输入张量
                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    # 6. 使用Stable Diffusion XL管道生成图像
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth=garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    # 7. 后处理 - 处理裁剪情况并返回结果
    if is_checked_crop:
        # 将生成的图片和mask调整到裁剪后的尺寸
        return images[0].resize(crop_size), mask_gray.resize(crop_size)
    else:
        # 直接将生成的图片和mask调整到原始尺寸
        return images[0].resize(orig_size), mask_gray.resize(orig_size)

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:

##文字標題所在

    gr.Markdown("## Change Clothes AI - AI Clothes Changer Online")
    gr.Markdown("Go to [Change Clothes AI](https://changeclothesai.online/) for Free Try-On! 🤗 .")
##係數區塊
    with gr.Column():
        try_button = gr.Button(value="Run Change Clothes AI")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)

##更衣區塊
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
            with gr.Row():
                category = gr.Dropdown(
                    choices=["upper_body", "lower_body", "dresses"],
                    label="Category",
                    value="upper_body"
                )
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=15,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(label="Description of garment", placeholder="Short Sleeve Round Neck T-shirts", show_label=True, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=16,
                examples=garm_list_path)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)

    with gr.Row():
        gr.Markdown("## Links")
        gr.Markdown("###### [Image Describer](http://imagedescriber.online/)")
        gr.Markdown("###### [Picture To Summary AI](https://picturetosummaryai.online/)")
        gr.Markdown("###### [PS2 Filter AI](https://ps2filterai.online/)")
        gr.Markdown("###### [Change Clothes AI](https://changeclothesai.online/)")
        gr.Markdown("###### [Describe Image AI](https://describeimageai.online/)")
        gr.Markdown("###### [Dress Changer AI Online](https://dresschangerai.online/)")
        gr.Markdown("###### [Image Extender AI](https://expandirimagenconia.online/)")
        gr.Markdown("###### [AI Accent Detector](https://aiaccentdetector.online/)")

    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed, category], outputs=[image_out,masked_img], api_name='tryon')




image_blocks.launch()
