import gc
import os
from typing import Any

from diffusers import FluxPriorReduxPipeline, FluxPipeline
from fastapi import FastAPI, Body
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from pydantic_models import FluxRequest, FluxResponse
from utils import base64_to_image, image_to_base64

PIPELINE: FluxPipeline
LOADED: bool = False
dtype = torch.bfloat16
avernus_flux = FastAPI()


def load_flux_pipeline():
    global PIPELINE
    PIPELINE = FluxPipeline.from_pretrained("Meatfucker/Flux.1-dev-bnb-nf4",
                                            torch_dtype=dtype).to("cuda")

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def load_ip_adapters(strength):
    global PIPELINE
    try:
        PIPELINE.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                                 weight_name="ip_adapter.safetensors",
                                 image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
        PIPELINE.set_ip_adapter_scale(strength)

    except Exception as e:
        print(e)

def load_flux_loras(lora_name):
    global PIPELINE
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                PIPELINE.load_lora_weights(f"./loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        PIPELINE.set_adapters(lora_list)
    except Exception:
        pass

def generate_flux(prompt,
                  width,
                  height,
                  steps,
                  batch_size,
                  image=None,
                  strength=None,
                  lora_name=None,
                  ip_adapter_image=None,
                  ip_adapter_strength=None,
                  guidance_scale=None,
                  seed=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_flux_pipeline()
        LOADED = True
    kwargs = {"width": width if width is not None else 1024,
              "height": height if height is not None else 1024,
              "num_inference_steps": steps if steps is not None else 30,
              "num_images_per_prompt": batch_size if batch_size is not None else 4}
    strength = strength if strength is not None else 1.0
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)
    if image is not None:
            kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = get_redux_embeds(image, prompt, strength)
    else:
        kwargs["prompt"] = prompt
    if ip_adapter_image is not None:
        try:
            load_ip_adapters(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception:
            pass
    if lora_name is not None:
        load_flux_loras(lora_name)
    PIPELINE.enable_model_cpu_offload() # This has to be after the ip adapter load or else you'll have tensor location problems
    PIPELINE.enable_vae_slicing()
    try:
        images = PIPELINE(**kwargs).images
        if lora_name is not None:
            PIPELINE.unload_lora_weights()
        if ip_adapter_image is not None:
            PIPELINE.unload_ip_adapter()
    except Exception:
        raise
    return images

def get_redux_embeds(image, prompt, strength):
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained("Meatfucker/Flux.1-dev-bnb-nf4", subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer_2 = T5TokenizerFast.from_pretrained("Meatfucker/Flux.1-dev-bnb-nf4", subfolder="tokenizer_2", torch_dtype=dtype)
    redux_pipeline = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev",
                                                            text_encoder=text_encoder,
                                                            text_encoder_2=text_encoder_2,
                                                            tokenizer=tokenizer,
                                                            tokenizer_2=tokenizer_2,
                                                            torch_dtype=dtype).to("cuda")
    redux_embeds, redux_pooled_embeds = redux_pipeline(image=image,
                                                       prompt=prompt,
                                                       prompt_2=prompt,
                                                       prompt_embeds_scale=strength,
                                                       pooled_prompt_embeds_scale=strength,
                                                       return_dict=False)
    redux_pipeline.to("cpu")
    del redux_pipeline.text_encoder, redux_pipeline.text_encoder_2, redux_pipeline.tokenizer, redux_pipeline.tokenizer_2, redux_pipeline
    torch.cuda.empty_cache()
    gc.collect()
    return redux_embeds, redux_pooled_embeds

@avernus_flux.post("/flux_generate", response_model=FluxResponse)
def flux_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
                              "width": data.width,
                              "height": data.height,
                              "steps": data.steps,
                              "batch_size": data.batch_size}
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = generate_flux(**kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        return e
    return {"images": base64_images}

@avernus_flux.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_flux, host="0.0.0.0", port=6970, log_level="critical")