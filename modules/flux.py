from diffusers import (FluxPriorReduxPipeline, FluxInpaintPipeline, FluxFillPipeline, FluxKontextPipeline,
                       FluxPipeline)
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torch
import gc
import math
import os

dtype = torch.bfloat16


async def load_flux_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "flux":
        print("loading FluxPipeline")
        await avernus_pipeline.delete_pipeline()
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        generator = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev",
                                                 quantization_config=pipeline_quant_config,
                                                 torch_dtype=dtype).to("cuda")
        await avernus_pipeline.set_pipeline(generator, "flux")
    return avernus_pipeline

async def load_flux_inpaint_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "flux_inpaint":
        print("loading FluxInpaintPipeline")
        await avernus_pipeline.delete_pipeline()
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )

        generator = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                        quantization_config=pipeline_quant_config,
                                                        torch_dtype=dtype).to("cuda")
        await avernus_pipeline.set_pipeline(generator, "flux_inpaint")
    return avernus_pipeline

async def load_flux_fill_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "flux_fill":
        print("loading FluxFillPipeline")
        await avernus_pipeline.delete_pipeline()
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        generator = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",
                                                     quantization_config=pipeline_quant_config,
                                                     torch_dtype=dtype).to("cuda")
        await avernus_pipeline.set_pipeline(generator, "flux_fill")
    return avernus_pipeline

async def load_flux_kontext_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "flux_kontext":
        print("loading FluxKontextPipeline")
        await avernus_pipeline.delete_pipeline()
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        generator = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                                        quantization_config=pipeline_quant_config,
                                                        torch_dtype=dtype).to("cuda")
        await avernus_pipeline.set_pipeline(generator, "flux_kontext")
    return avernus_pipeline

async def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

async def load_ip_adapters(avernus_pipeline, strength):
    try:
        avernus_pipeline.pipeline.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                                                  weight_name="ip_adapter.safetensors",
                                                  image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
        avernus_pipeline.pipeline.set_ip_adapter_scale(strength)
        return avernus_pipeline
    except Exception as e:
        print(f"FLUX IP ADAPTER ERROR: {e}")

async def load_flux_loras(avernus_pipeline, lora_name):
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                avernus_pipeline.pipeline.load_lora_weights(f"loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        avernus_pipeline.pipeline.set_adapters(lora_list)
        return avernus_pipeline
    except Exception as e:
        print(f"FLUX LORA ERROR: {e}")


async def generate_flux(avernus_pipeline,
                        prompt,
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

    kwargs = {}
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 1.0
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)

    if image is not None:
        if avernus_pipeline.model_type == "flux":
            avernus_pipeline.pipeline.to("cpu")
            kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = await get_redux_embeds(image, prompt, strength)
            avernus_pipeline.pipeline.to("cuda")
        else:
            await avernus_pipeline.delete_pipeline()
            kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = await get_redux_embeds(image, prompt, strength)
    else:
        kwargs["prompt"] = prompt

    avernus_pipeline = await load_flux_pipeline(avernus_pipeline)

    if ip_adapter_image is not None:
        try:
            avernus_pipeline = await load_ip_adapters(avernus_pipeline, ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"FLUX IP ADAPTER ERROR: {e}")

    if lora_name is not None:
        avernus_pipeline = await load_flux_loras(avernus_pipeline, lora_name)

    avernus_pipeline.pipeline.enable_vae_slicing()

    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
        if ip_adapter_image is not None:
            avernus_pipeline.pipeline.unload_ip_adapter()
    except Exception as e:
        print(f"Flux GENERATE ERROR: {e}")

    return images


async def generate_flux_inpaint(avernus_pipeline,
                                prompt,
                                width,
                                height,
                                steps,
                                batch_size,
                                image=None,
                                mask_image=None,
                                strength=None,
                                lora_name=None,
                                guidance_scale=None,
                                seed=None):
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

    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)

    avernus_pipeline = await load_flux_inpaint_pipeline(avernus_pipeline)

    if lora_name is not None:
        avernus_pipeline = await load_flux_loras(avernus_pipeline, lora_name)

    avernus_pipeline.pipeline.enable_vae_slicing()
    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
    except Exception as e:
        print(f"FLUX INPAINT GENERATE ERROR: {e}")

    return images


async def generate_flux_fill(avernus_pipeline,
                             prompt,
                             width,
                             height,
                             steps,
                             batch_size,
                             image=None,
                             mask_image=None,
                             strength=None,
                             lora_name=None,
                             guidance_scale=None,
                             seed=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["strength"] = strength if strength is not None else 0.9
    kwargs["image"] = image
    kwargs["mask_image"] = mask_image
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 30.0

    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)

    avernus_pipeline = await load_flux_fill_pipeline(avernus_pipeline)

    if lora_name is not None:
        avernus_pipeline = await load_flux_loras(avernus_pipeline, lora_name)

    avernus_pipeline.pipeline.enable_vae_slicing()
    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
    except Exception as e:
        print(f"FLUX FILL GENERATE ERROR: {e}")

    return images


async def generate_flux_kontext(avernus_pipeline,
                                prompt,
                                width,
                                height,
                                steps,
                                batch_size,
                                image,
                                lora_name=None,
                                ip_adapter_image=None,
                                ip_adapter_strength=None,
                                guidance_scale=None,
                                seed=None):
    kontext_width, kontext_height = resize_by_pixels(image.width, image.height)
    kwargs = {}
    kwargs["prompt"] = prompt
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = kontext_width
    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = kontext_height
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 1
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5
    kwargs["image"] = image
    if seed is not None:
        kwargs["generator"] = get_seed_generators(kwargs["num_images_per_prompt"], seed)

    avernus_pipeline = await load_flux_kontext_pipeline(avernus_pipeline)

    if ip_adapter_image is not None:
        try:
            avernus_pipeline = await load_ip_adapters(avernus_pipeline, ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"FLUX KONTEXT IP ADAPTER ERROR: {e}")

    if lora_name is not None:
        avernus_pipeline = await load_flux_loras(avernus_pipeline, lora_name)

    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
    except Exception as e:
        print(f"FLUX KONTEXT GENERATE ERROR: {e}")

    return images


async def get_redux_embeds(image, prompt, strength):
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer_2 = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2", torch_dtype=dtype)
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

def resize_by_pixels(width, height, target_pixels=1024*1024, keep_if_within=0.0):
    """
    Return (new_width, new_height) so total pixels ~= target_pixels,
    preserving aspect ratio. If current pixels are within ±keep_if_within
    (e.g. 0.25 for 25%), the original size is returned.
    """
    current = width * height
    if keep_if_within > 0 and abs(current - target_pixels) / target_pixels <= keep_if_within:
        return width, height

    scale = math.sqrt(target_pixels / current)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h