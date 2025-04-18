from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, utils
import torch
import gc
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
utils.logging.disable_progress_bar()


async def generate_sdxl(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        image=None,
                        strength=None,
                        negative_prompt=None,
                        model_name=None,
                        lora_name=None):
    width = width if width is not None else 1024
    height = height if height is not None else 1024
    steps = steps if steps is not None else 30
    batch_size = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 0.7
    if model_name is None:
        model_name = "misri/zavychromaxl_v100"

    if image is not None:
        generator = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name,
                                                                     torch_dtype=torch.float16,
                                                                     use_safetensors=True).to("cuda")
    else:
        generator = StableDiffusionXLPipeline.from_pretrained(model_name,
                                                              torch_dtype=torch.float16,
                                                              use_safetensors=True).to("cuda")
    if lora_name is not None:
        try:
            generator.load_lora_weights(f"loras/sdxl/{lora_name}", weight_name=lora_name)
        except Exception as e:
            print(f"SDXL LORA ERROR: {e}")
    generator.enable_model_cpu_offload()
    generator.set_progress_bar_config(disable=True)
    if image is not None:
        images = generator(prompt=prompt,
                           image=image,
                           negative_prompt=negative_prompt,
                           width=width, height=height,
                           num_inference_steps=steps,
                           strength=strength,
                           num_images_per_prompt=batch_size).images
    else:
        images = generator(prompt=prompt,
                           negative_prompt=negative_prompt,
                           width=width, height=height,
                           num_inference_steps=steps,
                           num_images_per_prompt=batch_size).images
    generator.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

