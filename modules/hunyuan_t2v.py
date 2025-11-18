from typing import Any
import tempfile

from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
import torch

from pydantic_models import HunyuanTI2VRequest
from utils import get_seed_generators, load_loras

PIPELINE: HunyuanVideoPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_hunyuan_t2v = FastAPI()


def load_hunyuan_pipeline(model_name="Meatfucker/HunyuanVideo-bnb-nf4", flow_shift=3.0):
    global PIPELINE
    PIPELINE = HunyuanVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_tiling()

def generate_hunyuan_ti2v(prompt: str,
                          video = None,
                          negative_prompt: str = None,
                          num_frames: int = 81,
                          guidance_scale: float = 5.0,
                          flow_shift: float = 3.0,
                          height: int = None,
                          width: int = None,
                          seed: int = None,
                          steps: int = 50,
                          model_name: str = None,
                          lora_name = None):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "Meatfucker/HunyuanVideo-bnb-nf4"
    if not LOADED:
        load_hunyuan_pipeline(model_name, flow_shift)
        LOADED = True
    kwargs:dict[str, Any] = {"prompt": prompt,
                             "negative_prompt": negative_prompt if negative_prompt is not None else "",
                             "num_frames": num_frames,
                             "guidance_scale": guidance_scale,
                             "height": height,
                             "width": width,
                             "num_inference_steps": steps}
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
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "hunyuan", lora_name)
    try:
        output = PIPELINE(**kwargs).frames[0]

    except Exception as e:
        return {"status": False,
                "status_message": e}
    return {"status": True,
            "video": output}


@avernus_hunyuan_t2v.post("/hunyuan_ti2v_generate")
def hunyuan_ti2v_generate(data: HunyuanTI2VRequest = Body(...)):
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
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    try:
        response = generate_hunyuan_ti2v(**kwargs)
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

@avernus_hunyuan_t2v.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_hunyuan_t2v, host="0.0.0.0", port=6970, log_level="critical")