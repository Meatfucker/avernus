from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline, utils, FluxImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from optimum.quanto import freeze, qfloat8, quantize
import torch
import gc


async def generate_flux(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        image=None,
                        strength=None,
                        model_name=None,
                        lora_name=None,
                        revision=None):
    width = width if width is not None else 1024
    height = height if height is not None else 1024
    steps = steps if steps is not None else 30
    batch_size = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 0.7

    generator = await get_pipeline(image, model_name, revision)
    if lora_name is not None:
        try:
            generator.load_lora_weights(f"loras/flux/{lora_name}", weight_name=lora_name)
        except Exception as e:
            print(f"FLUX LORA ERROR: {e}")
    generator = await quantize_components(generator)
    generator.to("cuda")
    generator.enable_model_cpu_offload()
    generator.set_progress_bar_config(disable=True)
    if image is not None:
        images = generator(prompt=prompt,
                           image=image,
                           width=width, height=height,
                           num_inference_steps=steps,
                           strength=strength,
                           num_images_per_prompt=batch_size).images
    else:
        images = generator(prompt=prompt,
                           width=width, height=height,
                           num_inference_steps=steps,
                           num_images_per_prompt=batch_size).images
    generator.to("cpu")

    del generator.scheduler, generator.text_encoder, generator.text_encoder_2, generator.tokenizer, generator.tokenizer_2, generator.vae, generator.transformer, generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

async def quantize_components(generator):
    quantize(generator.transformer, weights=qfloat8)
    freeze(generator.transformer)
    quantize(generator.text_encoder, weights=qfloat8)
    freeze(generator.text_encoder)
    quantize(generator.text_encoder_2, weights=qfloat8)
    freeze(generator.text_encoder_2)
    quantize(generator.vae, weights=qfloat8)
    freeze(generator.vae)
    return generator

async def get_pipeline(image, model_name, revision):
    if model_name is None:
        model_name = "black-forest-labs/FLUX.1-dev"
        revision = "refs/pr/3"
    dtype = torch.bfloat16
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=dtype,
                                                    revision=revision)
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2", torch_dtype=dtype,
                                                  revision=revision)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype, revision=revision)

    if image is not None:
        generator = FluxImg2ImgPipeline(scheduler=scheduler,
                                        text_encoder=text_encoder,
                                        tokenizer=tokenizer,
                                        text_encoder_2=text_encoder_2,
                                        tokenizer_2=tokenizer_2,
                                        vae=vae,
                                        transformer=transformer)
        return generator
    else:
        generator = FluxPipeline(scheduler=scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer,
                                 text_encoder_2=text_encoder_2,
                                 tokenizer_2=tokenizer_2,
                                 vae=vae,
                                 transformer=transformer)
        return generator