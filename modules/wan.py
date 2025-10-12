import os
from typing import Any
import tempfile

from diffusers import WanPipeline, UniPCMultistepScheduler, AutoencoderKLWan
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
import torch

from pydantic_models import WanTI2VRequest

PIPELINE: WanPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_wan = FastAPI()


def load_wan_pipeline(model_name="Meatfucker/Wan2.2-TI2V-5B-bnb-nf4", flow_shift=3.0):
    global PIPELINE
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    PIPELINE = WanPipeline.from_pretrained(model_name, vae=vae, torch_dtype=torch.bfloat16)
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config, flow_shift=flow_shift)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def generate_wan_v2v(prompt: str,
                     video = None,
                     negative_prompt: str = None,
                     num_frames: int = 81,
                     guidance_scale: float = 5.0,
                     flow_shift: float = 3.0,
                     height: int = None,
                     width: int = None,
                     seed: int = None,
                     model_name: str = None):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "Meatfucker/Wan2.2-TI2V-5B-bnb-nf4"
    if not LOADED:
        load_wan_pipeline(model_name, flow_shift)
        LOADED = True
    kwargs:dict[str, Any] = {"prompt": prompt,
                             "negative_prompt": negative_prompt if negative_prompt is not None else "",
                             "num_frames": num_frames,
                             "guidance_scale": guidance_scale,
                             "height": height,
                             "width": width}
    if seed is not None:
        kwargs["generator"] = get_seed_generators(1, seed)
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = 832

    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = 480
    try:
        output = PIPELINE(**kwargs).frames[0]
        return {"status": True,
                "video": output}
    except Exception as e:
        return {"status": False,
                "status_message": e}


@avernus_wan.post("/wan_ti2v_generate")
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


@avernus_wan.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_wan, host="0.0.0.0", port=6970, log_level="critical")