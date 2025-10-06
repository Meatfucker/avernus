import os
import tempfile
from typing import Any

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
import torch

from pydantic_models import WanTI2VRequest
from utils import base64_to_image, resize_by_pixels

PIPELINE: WanImageToVideoPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_wan_i2v = FastAPI()


def load_wan_pipeline(model_name="Meatfucker/Wan2.2-TI2V-5B-bnb-nf4"):
    global PIPELINE
    PIPELINE = WanImageToVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cpu")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def generate_wan_ti2v(prompt: str,
                     image = None,
                     negative_prompt: str = None,
                     num_frames: int = 81,
                     guidance_scale: float = 5.0,
                     height: int = None,
                     width: int = None,
                     seed: int = None,
                     model_name: str = None):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "Meatfucker/Wan2.2-TI2V-5B-bnb-nf4"
    if not LOADED:
        load_wan_pipeline(model_name)
        LOADED = True
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["num_frames"] = num_frames
    kwargs["guidance_scale"] = guidance_scale
    kwargs["height"] = height
    kwargs["width"] = width
    if seed is not None:
        kwargs["generator"] = get_seed_generators(1, seed)
    image_width, image_height = resize_by_pixels(image.width, image.height)
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = image_width
    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = image_height
    kwargs["image"] = image
    try:
        output = PIPELINE(**kwargs).frames[0]
        return {"status": True,
                "video": output}
    except Exception as e:
        return {"status": False,
                "status_message": e}


@avernus_wan_i2v.post("/wan_ti2v_generate")
def wan_ti2v_generate(data: WanTI2VRequest = Body(...)):
    kwargs: dict[str, Any] = {"prompt": data.prompt}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.width:
        kwargs["width"] = data.width
    if data.height:
        kwargs["height"] = data.height
    if data.steps:
        kwargs["steps"] = data.steps
    if data.num_frames:
        kwargs["num_frames"] = data.num_frames
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.model_name:
        kwargs["model_name"] = data.model_name
    try:
        response = generate_wan_ti2v(**kwargs)
        if response["status"] is True:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_path = tmp_file.name
            tmp_file.close()
            export_to_video(response["video"], tmp_path, fps=24)
            return {"status": True,
                    "path": tmp_path}
        else:
            return {"status": False,
                   "status_message": response["status_message"]}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}


@avernus_wan_i2v.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_wan_i2v, host="0.0.0.0", port=6970)