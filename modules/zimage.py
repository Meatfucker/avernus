from typing import Any

from diffusers import ZImagePipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import ZImageRequest, ImageResponse
from utils import image_to_base64, get_seed_generators, load_loras

PIPELINE: ZImagePipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_zimage = FastAPI()


def load_zimage_pipeline(model_name="Tongyi-MAI/Z-Image-Turbo"):
    global PIPELINE
    PIPELINE = ZImagePipeline.from_pretrained(model_name, torch_dtype=dtype).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_tiling()

def generate_zimage(prompt,
                    width,
                    height,
                    steps,
                    batch_size,
                    negative_prompt=None,
                    guidance_scale=None,
                    seed=None,
                    model_name=None,
                    lora_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        if model_name is not None:
            load_zimage_pipeline(model_name)
        else:
            load_zimage_pipeline()
        LOADED = True
    kwargs = {"prompt": prompt,
              "negative_prompt": negative_prompt if negative_prompt is not None else "",
              "width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 9,
              "num_images_per_prompt": batch_size if batch_size is not None else 4,
              "guidance_scale": guidance_scale if guidance_scale is not None else 0.0}
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "zimage", lora_name)
    try:
        images = PIPELINE(**kwargs).images
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_zimage.post("/zimage_generate", response_model=ImageResponse)
def zimage_generate(data: ZImageRequest = Body(...)):
    """Generates some number of ZImage images based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    try:
        response = generate_zimage(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "ZImage Success",
            "images": base64_images}

@avernus_zimage.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_zimage, host="0.0.0.0", port=6970, log_level="critical")