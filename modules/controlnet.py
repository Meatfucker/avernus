import cv2
from diffusers import ControlNetModel
import numpy
from PIL import Image
import torch
from transformers import pipeline
async def get_sdxl_controlnet(controlnet_processor, controlnet_image):
    if controlnet_processor == "canny":
        canny_image = await get_canny_image(controlnet_image)
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        return controlnet, canny_image
    if controlnet_processor == "depth":
        depth_image = await get_depth_image(controlnet_image)
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)
        return controlnet, depth_image

async def process_flux_image(controlnet_processor, controlnet_image):
    if controlnet_processor == "canny":
        canny_image = await get_canny_image(controlnet_image)
        return canny_image
    if controlnet_processor == "depth":
        depth_image = await get_depth_image(controlnet_image)
        return depth_image

async def get_canny_image(image):
    image = numpy.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

async def get_depth_image(image):
    model = "depth-anything/Depth-Anything-V2-base-hf"
    pipe = pipeline("depth-estimation", model=model, device=torch.device("cuda"))
    depth_keys = pipe(image)
    depth_image = depth_keys["depth"]
    return depth_image
