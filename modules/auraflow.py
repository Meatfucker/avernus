from typing import Any

from diffusers import AuraFlowPipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import AuraFlowRequest, ImageResponse
from utils import image_to_base64, get_seed_generators, load_loras

PIPELINE: AuraFlowPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_auraflow = FastAPI()


def load_auraflow_pipeline(model_name="fal/AuraFlow"):
    global PIPELINE
    PIPELINE = AuraFlowPipeline.from_pretrained(model_name, torch_dtype=dtype, safety_checker=None).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def generate_auraflow(prompt,
                      width,
                      height,
                      steps,
                      batch_size,
                      negative_prompt=None,
                      guidance_scale=None,
                      seed=None,
                      model_name="fal/AuraFlow",
                      lora_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_auraflow_pipeline(model_name)
        LOADED = True
    kwargs = {"prompt": prompt,
              "width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 30,
              "num_images_per_prompt": batch_size if batch_size is not None else 4,
              "guidance_scale": guidance_scale if guidance_scale is not None else 3.5}
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "auraflow", lora_name)
    try:
        images = PIPELINE(**kwargs).images
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_auraflow.post("/auraflow_generate", response_model=ImageResponse)
def auraflow_generate(data: AuraFlowRequest = Body(...)):
    """Generates some number of HiDream images based on user inputs"""
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
        response = generate_auraflow(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Auraflow Success",
            "images": base64_images}

@avernus_auraflow.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_auraflow, host="0.0.0.0", port=6970, log_level="critical")