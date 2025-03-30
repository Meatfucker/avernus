import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

async def generate_wan_i2v_480(user_image, user_prompt=None, user_negative_prompt=None):

    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    image = load_image(user_image)
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    prompt = user_prompt if user_prompt is not None else ""
    negative_prompt = user_negative_prompt if user_negative_prompt is not None else ""
    output = pipe(
        image=image, prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_frames=33, guidance_scale=5.0
    ).frames[0]
    # export_to_video(output, "output.mp4", fps=16)
    return output
