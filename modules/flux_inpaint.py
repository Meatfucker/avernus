import os
from typing import Any

from diffusers import FluxInpaintPipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import FluxInpaintRequest, ImageResponse
from utils import base64_to_image, image_to_base64

PIPELINE: FluxInpaintPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_flux_inpaint = FastAPI()

def load_flux_inpaint_pipeline(model_name="Meatfucker/Flux.1-dev-bnb-nf4"):
    global PIPELINE
    PIPELINE = FluxInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def load_flux_loras(lora_name):
    global PIPELINE
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                PIPELINE.load_lora_weights(f"./loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception:
                pass
        PIPELINE.set_adapters(lora_list)
    except Exception:
        pass

def generate_flux_inpaint(prompt,
                          width,
                          height,
                          steps,
                          batch_size,
                          negative_prompt=None,
                          image=None,
                          mask_image=None,
                          strength=None,
                          lora_name=None,
                          guidance_scale=None,
                          true_cfg_scale=None,
                          seed=None,
                          model_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        if model_name is not None:
            load_flux_inpaint_pipeline(model_name)
        else:
            load_flux_inpaint_pipeline()
        LOADED = True
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["strength"] = strength if strength is not None else 0.9
    kwargs["image"] = image
    kwargs["mask_image"] = mask_image
    kwargs["padding_mask_crop"] = 32
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 7.0
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 1.0
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if lora_name is not None:
        load_flux_loras(lora_name)
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_flux_inpaint.post("/flux_inpaint_generate", response_model=ImageResponse)
def flux_inpaint_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux inpaint images based on user inputs."""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.strength:
        kwargs["strength"] = data.strength
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.mask_image:
        kwargs["mask_image"] = base64_to_image(data.mask_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_flux_inpaint(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                    "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Flux Success",
            "images": base64_images}

@avernus_flux_inpaint.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_flux_inpaint, host="0.0.0.0", port=6970, log_level="critical")