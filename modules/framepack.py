import tempfile
from typing import Any

from diffusers import HunyuanVideoFramepackPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
import torch

from pydantic_models import FramepackRequest
from utils import base64_to_image

PIPELINE: HunyuanVideoFramepackPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_framepack = FastAPI()


def load_framepack_pipeline(model_name: str = "./models/FramepackI2V_HY"):
    global PIPELINE
    PIPELINE = HunyuanVideoFramepackPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    #apply_group_offloading(PIPELINE.text_encoder_2, onload_device=torch.device("cuda"), offload_type="block_level",
    #                       num_blocks_per_group=2)
    #PIPELINE.transformer.enable_group_offload(onload_device=torch.device("cuda"),
    #                              offload_device=torch.device("cpu"),
    #                              offload_type="leaf_level",
    #                              num_blocks_per_group=2)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    PIPELINE.vae.enable_tiling()
    

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def generate_framepack(prompt: str,
                       image = None,
                       last_image = None,
                       negative_prompt: str = None,
                       num_frames: int = 129,
                       guidance_scale: float = 6.0,
                       height: int = None,
                       width: int = None,
                       seed: int = None,
                       model_name: str = None):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "./models/FramepackI2V_HY"
    if not LOADED:
        load_framepack_pipeline(model_name)
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
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = 1280
    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = 720
    kwargs["image"] = image
    if last_image is not None:
        kwargs["last_image"] = last_image

    try:
        print(f"KWARGS:{kwargs}")
        output = PIPELINE(**kwargs).frames[0]
        return {"status": True,
                "video": output}
    except Exception as e:
        return {"status": False,
                "status_message": e}


@avernus_framepack.post("/framepack_generate")
def framepack_generate(data: FramepackRequest = Body(...)):
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
    if data.last_image:
        kwargs["last_image"] = base64_to_image(data.last_image)
    if data.model_name:
        kwargs["model_name"] = data.model_name
    try:
        response = generate_framepack(**kwargs)
        if response["status"] is True:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_path = tmp_file.name
            tmp_file.close()
            export_to_video(response["video"], tmp_path, fps=24)
            return {"status": True,
                    "path": tmp_path}
        else:
            return {"status": False,
                   "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}


@avernus_framepack.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(avernus_framepack, host="0.0.0.0", port=6970)
    #uvicorn.run(avernus_framepack, host="0.0.0.0", port=6970, log_level="critical")