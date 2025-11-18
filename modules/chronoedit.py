import os
from typing import Any

from diffusers.schedulers import UniPCMultistepScheduler
from fastapi import FastAPI, Body
import numpy as np
from PIL import Image
import torch
from transformers import CLIPVisionModel

from ChronoEdit.chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from pydantic_models import ChronoEditRequest, ImageResponse
from utils import base64_to_image, image_to_base64, get_seed_generators, load_loras

PIPELINE: ChronoEditPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_chronoedit = FastAPI()

def calculate_dimensions(image, mod_value):
    """
    Calculate output dimensions based on resolution settings.
    Args:
        image: PIL Image
        mod_value: Modulo value for dimension alignment
    Returns:
        Tuple of (width, height)
    """

    # Get max area from preset or override
    target_area = 720 * 1280

    # Calculate dimensions maintaining aspect ratio
    aspect_ratio = image.height / image.width
    calculated_height = (round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value)
    calculated_width = (round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value)

    return calculated_width, calculated_height

def load_chronoedit_pipeline(model_name="Meatfucker/ChronoEdit-bnb-nf4", flow_shift=2.0):
    global PIPELINE
    image_encoder = CLIPVisionModel.from_pretrained(model_name, subfolder="image_encoder", torch_dtype=torch.float32)
    PIPELINE = ChronoEditPipeline.from_pretrained(model_name, image_encoder=image_encoder, torch_dtype=dtype)
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config, flow_shift=flow_shift)
    PIPELINE.enable_model_cpu_offload()

def generate_chronoedit_image(prompt,
                              width,
                              height,
                              steps,
                              batch_size,
                              negative_prompt=None,
                              image=None,
                              lora_name=None,
                              guidance_scale=None,
                              flow_shift=2.0,
                              num_frames=None,
                              model_name="Meatfucker/ChronoEdit-bnb-nf4",
                              seed=None):
    try:
        global PIPELINE
        global LOADED
        if not LOADED:
            load_chronoedit_pipeline(model_name, flow_shift)
            LOADED = True
        mod_value = PIPELINE.vae_scale_factor_spatial * PIPELINE.transformer.config.patch_size[1]
        width, height = calculate_dimensions(image, mod_value)
        image = image.resize((width, height))
        kwargs = {}
        kwargs["prompt"] = prompt
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        kwargs["num_inference_steps"] = steps if steps is not None else 30
        kwargs["num_videos_per_prompt"] = batch_size if batch_size is not None else 1
        kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 1.0
        kwargs["num_frames"] = num_frames if num_frames is not None else 5
        if seed is not None:
            kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
        kwargs["image"] = image
        kwargs["width"] = width
        kwargs["height"] = height
        if lora_name is not None:
            PIPELINE = load_loras(PIPELINE, "chronoedit", lora_name)

        frames = PIPELINE(**kwargs).frames
        images = []
        for frame in frames:
            last_frame = (frame[-1] * 255).clip(0, 255).astype("uint8")
            image = Image.fromarray(last_frame)
            images.append(image)
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_chronoedit.post("/chronoedit_generate", response_model=ImageResponse)
def chronoedit_generate(data: ChronoEditRequest = Body(...)):
    """Generates some number of ChronoEdit images based on user inputs"""
    try:
        kwargs: dict[str, Any] = {"prompt": data.prompt,
                                  "width": data.width,
                                  "height": data.height,
                                  "steps": data.steps,
                                  "batch_size": data.batch_size}
        if isinstance(data.lora_name, str):
            kwargs["lora_name"] = [data.lora_name]
        else:
            kwargs["lora_name"] = data.lora_name
        if data.model_name:
            kwargs["model_name"] = data.model_name
        if data.negative_prompt:
            kwargs["negative_prompt"] = data.negative_prompt
        if data.image:
            kwargs["image"] = base64_to_image(data.image)
        if data.guidance_scale:
            kwargs["guidance_scale"] = data.guidance_scale
        if data.flow_shift:
            kwargs["flow_shift"] = data.flow_shift
        if data.num_frames:
            kwargs["num_frames"] = data.num_frames
        if data.seed:
            kwargs["seed"] = data.seed

        response = generate_chronoedit_image(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
            response = None
            del response
        else:
            return {"status": False,
                    "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "ChronoEdit Plus Success",
            "images": base64_images}

@avernus_chronoedit.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(avernus_chronoedit, host="0.0.0.0", port=6970)
    uvicorn.run(avernus_chronoedit, host="0.0.0.0", port=6970, log_level="critical")