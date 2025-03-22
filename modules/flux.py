from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline, utils
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from optimum.quanto import freeze, qfloat8, quantize
import torch
import gc
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
utils.logging.disable_progress_bar()


async def generate_flux(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        model_name=None,
                        ):
    width = width if width is not None else 1024
    height = height if height is not None else 1024
    steps = steps if steps is not None else 30
    batch_size = batch_size if batch_size is not None else 4
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
    transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype,
                                                         revision=revision)

    quantize(transformer, weights=qfloat8)
    freeze(transformer)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    generator = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    )
    generator.to("cuda")
    generator.enable_model_cpu_offload()
    generator.set_progress_bar_config(disable=True)
    images = generator(prompt=prompt,
                       width=width, height=height,
                       num_inference_steps=steps,
                       num_images_per_prompt=batch_size).images
    generator.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

async def generate_lora_flux(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        model_name=None,
                        lora_name=None
                        ):
    width = width if width is not None else 1024
    height = height if height is not None else 1024
    steps = steps if steps is not None else 30
    batch_size = batch_size if batch_size is not None else 1
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
    transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype,
                                                         revision=revision)

    quantize(transformer, weights=qfloat8)
    freeze(transformer)
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)

    generator = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    )
    generator.load_lora_weights(f"loras/flux/{lora_name}", weight_name=lora_name)
    generator.to("cuda")
    generator.enable_model_cpu_offload()
    generator.set_progress_bar_config(disable=True)
    images = generator(prompt=prompt,
                       width=width, height=height,
                       num_inference_steps=steps,
                       num_images_per_prompt=batch_size).images
    generator.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images