from typing import Any
import tempfile

from diffusers import WanVideoToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body, Form, UploadFile, File
from fastapi.responses import StreamingResponse
import torch

from pydantic_models import WanTI2VRequest

PIPELINE: WanVideoToVideoPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_wan_v2v = FastAPI()


def load_wan_pipeline(model_name="Meatfucker/Wan2.2-TI2V-5B-bnb-nf4", flow_shift=3.0):
    global PIPELINE
    PIPELINE = WanVideoToVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
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
                     flow_shift = 3.0,
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


@avernus_wan_v2v.post("/wan_v2v_generate")
def wan_v2v_generate(prompt: str = Form(...),
                     negative_prompt: str | None = Form(None),
                     width: int | None = Form(None),
                     height: int | None = Form(None),
                     steps: int | None = Form(None),
                     num_frames: int | None = Form(None),
                     guidance_scale: float | None = Form(None),
                     flow_shift: float | None = Form(None),
                     seed: int | None = Form(None),
                     model_name: str | None = Form(None),
                     video: UploadFile | None = File(None)
):
    kwargs: dict[str, Any] = {"prompt": prompt}

    # Collect all optional parameters
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    if steps:
        kwargs["steps"] = steps
    if num_frames:
        kwargs["num_frames"] = num_frames
    if guidance_scale:
        kwargs["guidance_scale"] = guidance_scale
    if flow_shift:
        kwargs["flow_shift"] = flow_shift
    if seed:
        kwargs["seed"] = seed
    if model_name:
        kwargs["model_name"] = model_name

    if video:
        # Save uploaded video temporarily
#        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#        tmp_input.write(video.file.read())
#        tmp_input.close()
        kwargs["video"] = video  # Pass to model function

    try:
        response = generate_wan_v2v(**kwargs)
        if response["status"] is True:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_path = tmp_file.name
            tmp_file.close()
            export_to_video(response["video"], tmp_path, fps=24)
            return {"status": True, "path": tmp_path}
        else:
            return {"status": False, "status_message": response["status_message"]}
    except Exception as e:
        return {"status": False, "status_message": str(e)}


@avernus_wan_v2v.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_wan_v2v, host="0.0.0.0", port=6970, log_level="critical")