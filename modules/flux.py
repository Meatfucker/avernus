from diffusers import (FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline,
                       FluxControlPipeline, FluxControlNetModel, FluxPriorReduxPipeline, FluxInpaintPipeline)
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from optimum.quanto import freeze, qfloat8, quantize
import torch
import gc
from modules.controlnet import process_flux_image

dtype = torch.bfloat16


async def generate_flux(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        image=None,
                        strength=None,
                        model_name=None,
                        lora_name=None,
                        revision=None,
                        controlnet_processor=None,
                        controlnet_image=None,
                        ip_adapter_image=None,
                        ip_adapter_strength=None,
                        guidance_scale=None):
    kwargs = {}
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 1.0
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5

    if image is not None:
        redux_embeds, redux_pooled_embeds = await get_redux_embeds(image, prompt, strength)
        kwargs["prompt_embeds"] = redux_embeds
        kwargs["pooled_prompt_embeds"] = redux_pooled_embeds
    else:
        kwargs["prompt"] = prompt


    generator = await get_pipeline(image, model_name, revision, controlnet_processor)

    if ip_adapter_image is not None:
        try:
            generator.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                                      weight_name="ip_adapter.safetensors",
                                      image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
            generator.set_ip_adapter_scale(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"FLUX IP ADAPTER ERROR: {e}")
    if lora_name is not None:
        try:
            generator.load_lora_weights(f"loras/flux/{lora_name}", weight_name=lora_name)
        except Exception as e:
            print(f"FLUX LORA ERROR: {e}")
    if controlnet_processor is not None:
        processed_image = await process_flux_image(controlnet_processor, controlnet_image)
        kwargs["control_image"] = processed_image


    generator = await quantize_components(generator)
    generator.to("cuda")
    generator.enable_model_cpu_offload()
    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"Flux GENERATE ERROR: {e}")
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

async def get_pipeline(image, model_name, revision, controlnet_processor):
    scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer, controlnet = await get_components(controlnet_processor, image)

    if controlnet_processor is None:
        print("loading FluxPipeline")
        generator = FluxPipeline(scheduler=scheduler,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer,
                                 text_encoder_2=text_encoder_2,
                                 tokenizer_2=tokenizer_2,
                                 vae=vae,
                                 transformer=transformer)
    else:
        print("loading FluxControlPipeline")
        generator = FluxControlPipeline(scheduler=scheduler,
                                        text_encoder=text_encoder,
                                        tokenizer=tokenizer,
                                        text_encoder_2=text_encoder_2,
                                        tokenizer_2=tokenizer_2,
                                        vae=vae,
                                        transformer=transformer)
    return generator


async def get_components(controlnet_processor, image):
    if controlnet_processor is None:
        model_name = "black-forest-labs/FLUX.1-dev"
        revision = "refs/pr/3"
        text_encoder, tokenizer, text_encoder_2, tokenizer_2 = await get_text_encoders()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler",
                                                                    revision=revision)
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype,
                                                             revision=revision)
        return scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer, None

    if controlnet_processor == "depth":
        if image:
            controlnet = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=dtype)
            model_name = "black-forest-labs/FLUX.1-dev"
        else:
            controlnet = None
            model_name = "black-forest-labs/FLUX.1-Depth-dev"

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
        transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype)
        return scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer, controlnet

    if controlnet_processor == "canny":
        if image:
            controlnet = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=dtype)
            model_name = "black-forest-labs/FLUX.1-dev"
        else:
            controlnet = None
            model_name = "black-forest-labs/FLUX.1-Canny-dev"

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype)
        transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype)
        return scheduler, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, transformer, controlnet

async def get_redux_embeds(image, prompt, strength):
    redux_repo = "black-forest-labs/FLUX.1-Redux-dev"
    text_encoder, tokenizer, text_encoder_2, tokenizer_2 = await get_text_encoders()
    redux_pipeline = FluxPriorReduxPipeline.from_pretrained(redux_repo,
                                                            text_encoder=text_encoder,
                                                            tokenizer=tokenizer,
                                                            text_encoder_2=text_encoder_2,
                                                            tokenizer_2=tokenizer_2,
                                                            torch_dtype=dtype).to("cuda")
    redux_embeds, redux_pooled_embeds = redux_pipeline(image=image,
                                                       prompt=prompt,
                                                       prompt_2=prompt,
                                                       prompt_embeds_scale=strength,
                                                       pooled_prompt_embeds_scale=strength,
                                                       return_dict=False)
    redux_pipeline.to("cpu")
    del redux_pipeline, text_encoder, tokenizer, text_encoder_2, tokenizer_2
    torch.cuda.empty_cache()
    gc.collect()

    return redux_embeds, redux_pooled_embeds

async def get_text_encoders():
    model_name = "black-forest-labs/FLUX.1-dev"
    revision = "refs/pr/3"
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=dtype,
                                                    revision=revision)
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2", torch_dtype=dtype,
                                                  revision=revision)
    return text_encoder, tokenizer, text_encoder_2, tokenizer_2


async def generate_flux_inpaint(prompt,
                                width,
                                height,
                                steps,
                                batch_size,
                                image=None,
                                mask_image=None,
                                strength=None,
                                model_name=None,
                                lora_name=None,
                                revision=None,
                                guidance_scale=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["strength"] = strength if strength is not None else 0.9
    kwargs["image"] = image
    kwargs["mask_image"] = mask_image
    kwargs["padding_mask_crop"] = 32
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 7.0

    model_name = "black-forest-labs/FLUX.1-dev"
    revision = "refs/pr/3"
    text_encoder, tokenizer, text_encoder_2, tokenizer_2 = await get_text_encoders()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler",
                                                                revision=revision)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=dtype,
                                                         revision=revision)

    print("loading FluxInpaintPipeline")
    generator = FluxInpaintPipeline(scheduler=scheduler,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    text_encoder_2=text_encoder_2,
                                    tokenizer_2=tokenizer_2,
                                    vae=vae,
                                    transformer=transformer)

    if lora_name is not None:
        try:
            generator.load_lora_weights(f"loras/flux/{lora_name}", weight_name=lora_name)
        except Exception as e:
            print(f"FLUX LORA ERROR: {e}")

    generator = await quantize_components(generator)
    generator.to("cuda")
    generator.enable_model_cpu_offload()
    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"FLUX INPAINT GENERATE ERROR: {e}")
    generator.to("cpu")

    del text_encoder, tokenizer, text_encoder_2, tokenizer_2, scheduler, vae, transformer
    del generator.scheduler, generator.text_encoder, generator.text_encoder_2, generator.tokenizer, generator.tokenizer_2, generator.vae, generator.transformer, generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

