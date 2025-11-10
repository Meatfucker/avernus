from typing import Any

from diffusers import Lumina2Pipeline, Lumina2Transformer2DModel
from fastapi import FastAPI, Body
import torch

from pydantic_models import LuminaRequest, ImageResponse
from utils import image_to_base64

PIPELINE: Lumina2Pipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_lumina2 = FastAPI()


def load_lumina2_pipeline(model_name="Alpha-VLLM/Lumina-Image-2.0"):
    global PIPELINE
    if model_name.endswith("safetensors"):
        transformer = Lumina2Transformer2DModel.from_single_file(model_name, torch_dtype=dtype)
        PIPELINE = Lumina2Pipeline.from_pretrained("Alpha-VLLM/Lumina-Image-2.0",
                                                   transformer=transformer,
                                                   torch_dtype=dtype).to("cuda")
    else:
        PIPELINE = Lumina2Pipeline.from_pretrained(model_name, torch_dtype=dtype).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator


def generate_lumina2(prompt,
                     width,
                     height,
                     steps,
                     batch_size,
                     guidance_scale=None,
                     seed=None,
                     model_name="Alpha-VLLM/Lumina-Image-2.0"):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_lumina2_pipeline(model_name)
        LOADED = True
    kwargs = {"prompt": prompt,
              "width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 30,
              "num_images_per_prompt": batch_size if batch_size is not None else 4,
              "guidance_scale": guidance_scale if guidance_scale is not None else 4.0}
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    try:
        images = PIPELINE(**kwargs).images
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_lumina2.post("/lumina2_generate", response_model=ImageResponse)
def lumina2_generate(data: LuminaRequest = Body(...)):
    """Generates some number of Lumina2 images based on user inputs"""
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
        response = generate_lumina2(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Lumina2 Success",
            "images": base64_images}

@avernus_lumina2.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn

    #uvicorn.run(avernus_lumina2, host="0.0.0.0", port=6970)
    uvicorn.run(avernus_lumina2, host="0.0.0.0", port=6970, log_level="critical")