import gc
from typing import Any

from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from fastapi import FastAPI, Body
import torch
from transformers import Mistral3ForConditionalGeneration

from pydantic_models import FluxRequest, ImageResponse
from utils import base64_to_image, image_to_base64, get_seed_generators, load_loras

PIPELINE: Flux2Pipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_flux2 = FastAPI()


def load_flux2_pipeline(model_name="diffusers/FLUX.2-dev-bnb-4bit"):
    global PIPELINE
    transformer = Flux2Transformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype,
                                                          device_map="cpu")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(model_name, subfolder="text_encoder", dtype=dtype,
                                                                    device_map="cpu")
    PIPELINE = Flux2Pipeline.from_pretrained(
        model_name, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_tiling()


def generate_flux2(prompt,
                   width,
                   height,
                   steps,
                   batch_size,
                   negative_prompt=None,
                   image=None,
                   lora_name=None,
                   guidance_scale=None,
                   true_cfg_scale=None,
                   seed=None,
                   model_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        if model_name is not None:
            load_flux2_pipeline(model_name)
        else:
            load_flux2_pipeline()
        LOADED = True
    kwargs = {"width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 30,
              "num_images_per_prompt": batch_size if batch_size is not None else 4}
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 4
    #kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 1.0
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if image is not None:
        kwargs["image"] = image
    else:
        kwargs["prompt"] = prompt
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "flux2", lora_name)

    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}


@avernus_flux2.post("/flux2_generate", response_model=ImageResponse)
def flux2_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
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
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_flux2(**kwargs)
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

@avernus_flux2.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_flux2, host="0.0.0.0", port=6970, log_level="critical")