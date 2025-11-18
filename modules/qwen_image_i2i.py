from typing import Any

from diffusers import QwenImageImg2ImgPipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import QwenImageRequest, ImageResponse
from utils import base64_to_image, image_to_base64, get_seed_generators, load_loras

PIPELINE: QwenImageImg2ImgPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_qwen_image = FastAPI()

def load_qwen_image_pipeline():
    global PIPELINE
    PIPELINE = QwenImageImg2ImgPipeline.from_pretrained("Meatfucker/Qwen-Image-bnb-nf4", torch_dtype=dtype)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def generate_qwen_image(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        negative_prompt=None,
                        image=None,
                        strength=None,
                        lora_name=None,
                        true_cfg_scale=None,
                        seed=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_qwen_image_pipeline()
        LOADED = True
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 0.7
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 4.0
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    kwargs["image"] = image
    kwargs["strength"] = strength
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "qwen_image", lora_name)
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_qwen_image.post("/qwen_image_i2i_generate", response_model=ImageResponse)
def qwen_image_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image images based on user inputs"""
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
        kwargs["strength"] = data.strength
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_qwen_image(**kwargs)
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
            "status_message": "Qwen Image Edit Plus Success",
            "images": base64_images}

@avernus_qwen_image.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_qwen_image, host="0.0.0.0", port=6970, log_level="critical")