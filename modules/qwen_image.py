import os
from typing import Any

from diffusers import QwenImagePipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import QwenImageRequest, QwenImageResponse
from utils import image_to_base64

PIPELINE: QwenImagePipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_qwen_image = FastAPI()

def load_qwen_image_pipeline():
    global PIPELINE
    PIPELINE = QwenImagePipeline.from_pretrained("Meatfucker/Qwen-Image-bnb-nf4", torch_dtype=dtype)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def load_qwen_loras(lora_name):
    global PIPELINE
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                PIPELINE.load_lora_weights(f"loras/qwen/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception:
                pass
        PIPELINE.set_adapters(lora_list)
    except Exception:
        pass

def generate_qwen_image(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        negative_prompt=None,
                        lora_name=None,
                        true_cfg_scale=None,
                        seed=None):
    global PIPELINE
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 4.0
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    global LOADED
    if not LOADED:
        load_qwen_image_pipeline()
        LOADED = True
    if lora_name is not None:
        load_qwen_loras(lora_name)
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_qwen_image.post("/qwen_image_generate", response_model=QwenImageResponse)
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
                    "status_message": response["status_message"]}
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
    #uvicorn.run(avernus_qwen_image, host="0.0.0.0", port=6970)