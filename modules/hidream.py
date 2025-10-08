from typing import Any

from diffusers import HiDreamImagePipeline
from fastapi import FastAPI, Body
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from pydantic_models import HiDreamRequest, HiDreamResponse
from utils import base64_to_image, image_to_base64

PIPELINE: HiDreamImagePipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_hidream = FastAPI()


def load_hidream_pipeline(model_name="Meatfucker/HiDream-I1-Full-bnb-nf4"):
    global PIPELINE
    PIPELINE = HiDreamImagePipeline.from_pretrained(model_name, torch_dtype=dtype).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator


def generate_hidream(prompt,
                     width,
                     height,
                     steps,
                     batch_size,
                     guidance_scale=None,
                     seed=None,
                     model_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        if model_name is not None:
            load_hidream_pipeline(model_name)
        else:
            load_hidream_pipeline()
        LOADED = True
    kwargs = {"prompt": prompt,
              "width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 30,
              "num_images_per_prompt": batch_size if batch_size is not None else 4,
              "guidance_scale": guidance_scale if guidance_scale is not None else 5.0}
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    try:
        images = PIPELINE(**kwargs).images
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_hidream.post("/hidream_generate", response_model=HiDreamResponse)
def hidream_generate(data: HiDreamRequest = Body(...)):
    """Generates some number of HiDream images based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_hidream(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                "status_message": response["status_message"]}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "HiDream Success",
            "images": base64_images}

@avernus_hidream.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_hidream, host="0.0.0.0", port=6970)