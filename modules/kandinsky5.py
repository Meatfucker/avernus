from typing import Any
import tempfile

from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, Body
import torch

from pydantic_models import KandinskyT2VRequest

PIPELINE: Kandinsky5T2VPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_kandinsky5_t2v = FastAPI()


def load_kandinsky5_pipeline(model_name="ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"):
    global PIPELINE
    PIPELINE = Kandinsky5T2VPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_tiling()

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def generate_kandinsky5_t2v(prompt: str,
                            negative_prompt: str = None,
                            num_frames: int = 121,
                            guidance_scale: float = 5.0,
                            height: int = None,
                            width: int = None,
                            seed: int = None,
                            model_name: str = None,
                            steps = 50):
    global PIPELINE
    global LOADED
    if model_name is None:
        model_name = "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
    if not LOADED:
        load_kandinsky5_pipeline(model_name)
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
        kwargs["width"] = 768

    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = 512
    try:
        output = PIPELINE(**kwargs).frames[0]

    except Exception as e:
        return {"status": False,
                "status_message": e}
    return {"status": True,
            "video": output}


@avernus_kandinsky5_t2v.post("/kandinsky5_t2v_generate")
def kandinsky5_t2v_generate(data: KandinskyT2VRequest = Body(...)):
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
        response = generate_kandinsky5_t2v(**kwargs)
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

@avernus_kandinsky5_t2v.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_kandinsky5_t2v, host="0.0.0.0", port=6970)