import os
from typing import Any

from diffusers import FluxKontextPipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import FluxRequest, ImageResponse
from utils import base64_to_image, image_to_base64, resize_by_pixels, get_seed_generators, load_loras

PIPELINE: FluxKontextPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_flux_kontext = FastAPI()

def load_flux_kontext_pipeline():
    global PIPELINE
    PIPELINE = FluxKontextPipeline.from_pretrained("Meatfucker/Flux.1-Kontext-dev-bnb-nf4",
                                                    torch_dtype=dtype).to("cuda")

def load_ip_adapters(strength):
    global PIPELINE
    try:
        PIPELINE.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                                                  weight_name="ip_adapter.safetensors",
                                                  image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
        PIPELINE.set_ip_adapter_scale(strength)
    except Exception:
        pass

def generate_flux_kontext(prompt,
                          width,
                          height,
                          steps,
                          batch_size,
                          image,
                          negative_prompt=None,
                          lora_name=None,
                          ip_adapter_image=None,
                          ip_adapter_strength=None,
                          guidance_scale=None,
                          true_cfg_scale=None,
                          seed=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_flux_kontext_pipeline()
        LOADED = True
    kontext_width, kontext_height = resize_by_pixels(image.width, image.height)
    kwargs = {}
    kwargs["prompt"] = prompt
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = kontext_width
    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = kontext_height
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 1
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 1.0
    kwargs["image"] = image
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if ip_adapter_image is not None:
        try:
            load_ip_adapters(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception:
            pass
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "flux", lora_name)
    PIPELINE.enable_model_cpu_offload() # This has to be after the ip adapter load or else you'll have tensor location problems
    PIPELINE.vae.enable_slicing()
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_flux_kontext.post("/flux_kontext_generate", response_model=ImageResponse)
def flux_kontext_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux Kontext images based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_flux_kontext(**kwargs)
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

@avernus_flux_kontext.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_flux_kontext, host="0.0.0.0", port=6970, log_level="critical")