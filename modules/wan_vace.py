import tempfile
from typing import Any

from diffusers import WanVACEPipeline, UniPCMultistepScheduler, AutoencoderKLWan
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
import PIL.Image
import torch

from pydantic_models import WanVACERequest
from utils import base64_to_image, get_seed_generators, load_loras

PIPELINE: WanVACEPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_wan_vace = FastAPI()


def load_wan_pipeline(model_name="Meatfucker/Wan2.1-VACE-1.3B-nf4-bnb", flow_shift=3.0):
    global PIPELINE
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    PIPELINE = WanVACEPipeline.from_pretrained(model_name, vae=vae, torch_dtype=torch.bfloat16).to("cpu")
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(PIPELINE.scheduler.config, flow_shift=flow_shift)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_tiling()

def prepare_i2v_video_and_mask(img: PIL.Image.Image, height: int, width: int, num_frames: int):
    img = img.resize((width, height))
    frames = [img]
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 1))
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 1)]
    return frames, mask

def prepare_flf2v_video_and_mask(first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask

def prepare_v2lf_video_and_mask(img: PIL.Image.Image, height: int, width: int, num_frames: int):
    img = img.resize((width, height))
    frames = []
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 1))
    frames.append(img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [*[mask_white] * (num_frames - 1), mask_black]
    return frames, mask

def generate_wan_vace(prompt: str,
                      first_frame = None,
                      last_frame = None,
                      negative_prompt: str = None,
                      num_frames: int = 81,
                      guidance_scale: float = 5.0,
                      height: int = None,
                      width: int = None,
                      seed: int = None,
                      steps: int = 50,
                      model_name: str = None,
                      flow_shift = 3.0,
                      lora_name=None):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "Meatfucker/Wan2.1-VACE-1.3B-nf4-bnb"
    if not LOADED:
        load_wan_pipeline(model_name, flow_shift)
        LOADED = True
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["num_frames"] = num_frames
    kwargs["guidance_scale"] = guidance_scale
    kwargs["height"] = height
    kwargs["width"] = width
    kwargs["num_inference_steps"] = steps
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
    video = None
    mask = None
    try:
        if first_frame is not None and last_frame is not None:
            print("both frames")
            video, mask = prepare_flf2v_video_and_mask(first_frame, last_frame, height, width, num_frames)
            kwargs["video"] = video
            kwargs["mask"] = mask
        if last_frame is not None and first_frame is None:
            print("last frame")
            video, mask = prepare_v2lf_video_and_mask(last_frame, height, width, num_frames)
            kwargs["video"] = video
            kwargs["mask"] = mask
        if first_frame is not None and last_frame is None:
            print("first frame")
            video, mask = prepare_i2v_video_and_mask(first_frame, height, width, num_frames)
            kwargs["video"] = video
            kwargs["mask"] = mask

    except Exception as e:
        return {"status": False,
                "status_message": e}
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "wan", lora_name)
    try:
        output = PIPELINE(**kwargs).frames[0]
        return {"status": True,
                "video": output}
    except Exception as e:
        return {"status": False,
                "status_message": e}


@avernus_wan_vace.post("/wan_vace_generate")
def wan_vace_generate(data: WanVACERequest = Body(...)):
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
    if data.first_frame:
        kwargs["first_frame"] = base64_to_image(data.first_frame)
    if data.last_frame:
        kwargs["last_frame"] = base64_to_image(data.last_frame)
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    try:
        response = generate_wan_vace(**kwargs)
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


@avernus_wan_vace.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn

    #uvicorn.run(avernus_wan_vace, host="0.0.0.0", port=6970)
    uvicorn.run(avernus_wan_vace, host="0.0.0.0", port=6970, log_level="critical")