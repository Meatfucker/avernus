import os
from typing import Any

from diffusers import (StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler, DDPMScheduler,
                       LMSDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler,
                       EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler,
                       DPMSolverSDEScheduler, PNDMScheduler)
from fastapi import FastAPI, Body
import torch

from pydantic_models import SDXLRequest, ImageResponse
from utils import base64_to_image, image_to_base64

PIPELINE: StableDiffusionXLPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_sdxl = FastAPI()

def load_sdxl_pipeline(model_name):
    global PIPELINE
    PIPELINE = StableDiffusionXLPipeline.from_pretrained(model_name,
                                                         torch_dtype=dtype,
                                                         use_safetensors=True).to("cuda")
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def load_ip_adapters(strength):
    global PIPELINE
    try:
        PIPELINE.load_ip_adapter("h94/IP-Adapter",
                                 subfolder="sdxl_models",
                                 weight_name="ip-adapter_sdxl.bin",
                                 device="cuda")
        PIPELINE.set_ip_adapter_scale(strength)
    except Exception:
        pass

def load_sdxl_loras(lora_name):
    global PIPELINE
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                PIPELINE.load_lora_weights(f"loras/sdxl/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception:
                pass
        PIPELINE.set_adapters(lora_list)
    except Exception:
        pass

def generate_sdxl(prompt,
                  width,
                  height,
                  steps,
                  batch_size,
                  image=None,
                  strength=None,
                  negative_prompt=None,
                  model_name=None,
                  lora_name=None,
                  controlnet_processor=None,
                  controlnet_image=None,
                  controlnet_conditioning=None,
                  ip_adapter_image=None,
                  ip_adapter_strength=None,
                  guidance_scale=None,
                  scheduler=None,
                  seed=None):
    global PIPELINE
    if model_name is None:
        model_name = "misri/zavychromaxl_v100"
    global LOADED
    if not LOADED:
        load_sdxl_pipeline(model_name)
        LOADED = True
    kwargs = {}
    kwargs["prompt"] = prompt
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 5.0
    strength = strength if strength is not None else 0.7
    controlnet_conditioning = controlnet_conditioning if controlnet_conditioning is not None else 0.5
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if scheduler is not None:
        set_scheduler(scheduler)
    if ip_adapter_image is not None:
        try:
            load_ip_adapters(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception:
            pass
    if lora_name is not None:
        load_sdxl_loras(lora_name)
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        if ip_adapter_image is not None:
            PIPELINE.unload_ip_adapter()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

def set_scheduler(scheduler):
    global PIPELINE
    try:
        match scheduler:
            case "DDIMScheduler":
                PIPELINE.scheduler = DDIMScheduler.from_config(PIPELINE.scheduler.config)
            case "DDPMScheduler":
                PIPELINE.scheduler = DDPMScheduler.from_config(PIPELINE.scheduler.config)
            case "PNDMScheduler":
                PIPELINE.scheduler = PNDMScheduler.from_config(PIPELINE.scheduler.config)
            case "LMSDiscreteScheduler":
                PIPELINE.scheduler = LMSDiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "EulerDiscreteScheduler":
                PIPELINE.scheduler = EulerDiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "HeunDiscreteScheduler":
                PIPELINE.scheduler = HeunDiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "EulerAncestralDiscreteScheduler":
                PIPELINE.scheduler = EulerAncestralDiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "DPMSolverMultistepScheduler":
                PIPELINE.scheduler = DPMSolverMultistepScheduler.from_config(PIPELINE.scheduler.config)
            case "DPMSolverSinglestepScheduler":
                PIPELINE.scheduler = DPMSolverSinglestepScheduler.from_config(PIPELINE.scheduler.config)
            case "KDPM2DiscreteScheduler":
                PIPELINE.scheduler = KDPM2DiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "KDPM2AncestralDiscreteScheduler":
                PIPELINE.scheduler = KDPM2AncestralDiscreteScheduler.from_config(PIPELINE.scheduler.config)
            case "DEISMultistepScheduler":
                PIPELINE.scheduler = DEISMultistepScheduler.from_config(PIPELINE.scheduler.config)
            case "UniPCMultistepScheduler":
                PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config)
            case "DPMSolverSDEScheduler":
                PIPELINE.scheduler = DPMSolverSDEScheduler.from_config(PIPELINE.scheduler.config)
    except Exception:
        pass

@avernus_sdxl.post("/sdxl_generate", response_model=ImageResponse)
def sdxl_generate(data: SDXLRequest = Body(...)):
    """Generates some number of sdxl images based on user inputs."""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "negative_prompt": data.negative_prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size,
                              "model_name": data.model_name}
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.scheduler:
        kwargs["scheduler"] = data.scheduler
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_sdxl(**kwargs)
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
            "status_message": "SDXL Success",
            "images": base64_images}

@avernus_sdxl.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_sdxl, host="0.0.0.0", port=6970, log_level="critical")