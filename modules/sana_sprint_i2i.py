from typing import Any

from diffusers import SanaSprintImg2ImgPipeline
from fastapi import FastAPI, Body
import torch

from pydantic_models import SanaSprintRequest, ImageResponse
from utils import image_to_base64, base64_to_image, get_seed_generators, load_loras

PIPELINE: SanaSprintImg2ImgPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_sana_sprint = FastAPI()


def load_sana_sprint_pipeline(model_name="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"):
    global PIPELINE
    PIPELINE = SanaSprintImg2ImgPipeline.from_pretrained(model_name, torch_dtype=dtype).to("cuda")
    PIPELINE.enable_model_cpu_offload()
    PIPELINE.vae.enable_slicing()

def generate_sana_sprint(prompt,
                         width,
                         height,
                         steps,
                         batch_size,
                         image,
                         strength=None,
                         guidance_scale=None,
                         seed=None,
                         max_timesteps=None,
                         intermediate_timesteps=None,
                         model_name=None,
                         lora_name=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        if model_name is not None:
            load_sana_sprint_pipeline(model_name)
        else:
            load_sana_sprint_pipeline()
        LOADED = True
    kwargs = {"prompt": prompt,
              "image": image,
              "width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 2,
              "num_images_per_prompt": batch_size if batch_size is not None else 4,
              "guidance_scale": guidance_scale if guidance_scale is not None else 4.5,
              "strength": strength if strength is not None else 0.9}
    if max_timesteps is not None:
        kwargs["max_timesteps"] = max_timesteps
    if intermediate_timesteps is not None:
        kwargs["intermediate_timesteps"] = intermediate_timesteps
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if lora_name is not None:
        PIPELINE = load_loras(PIPELINE, "sana_sprint", lora_name)
    try:
        images = PIPELINE(**kwargs).images
        return {"status": True,
                "images": images}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_sana_sprint.post("/sana_sprint_generate", response_model=ImageResponse)
def sana_sprint_generate(data: SanaSprintRequest = Body(...)):
    """Generates some number of Sana Sprint images based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.max_timesteps:
        kwargs["max_timesteps"] = data.max_timesteps
    if data.intermediate_timesteps:
        kwargs["intermediate_timesteps"] = data.intermediate_timesteps
    if data.model_name:
        kwargs["model_name"] = data.model_name
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    if data.strength:
        kwargs["strength"] = data.strength
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    try:
        response = generate_sana_sprint(**kwargs)
        if response["status"] is True:
            base64_images = [image_to_base64(img) for img in response["images"]]
        else:
            return {"status": False,
                "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Sana Sprint I2I Success",
            "images": base64_images}

@avernus_sana_sprint.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_sana_sprint, host="0.0.0.0", port=6970, log_level="critical")