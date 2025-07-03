from diffusers import (FluxPriorReduxPipeline, FluxInpaintPipeline, FluxFillPipeline, FluxKontextPipeline,
                       FluxPipeline)
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torch
import gc
import os

dtype = torch.bfloat16

async def generate_flux(prompt,
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
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    if image is not None:
        redux_embeds, redux_pooled_embeds = await get_redux_embeds(image, prompt, strength)
        kwargs["prompt_embeds"] = redux_embeds
        kwargs["pooled_prompt_embeds"] = redux_pooled_embeds
    else:
        kwargs["prompt"] = prompt

    print("loading FluxPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )

    generator = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                             quantization_config=pipeline_quant_config,
                                             torch_dtype=dtype).to("cuda")

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
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        generator.set_adapters(lora_list)
        generator.fuse_lora(adapter_names=lora_list)
        generator.unload_lora_weights()

    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"Flux GENERATE ERROR: {e}")

    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images


async def generate_flux_inpaint(prompt,
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
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    print("loading FluxInpaintPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )

    generator = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                    quantization_config=pipeline_quant_config,
                                                    torch_dtype=dtype).to("cuda")

    if lora_name is not None:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        generator.set_adapters(lora_list)

    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"FLUX INPAINT GENERATE ERROR: {e}")

    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images


async def generate_flux_fill(prompt,
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
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    print("loading FluxFillPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",
                                                 quantization_config=pipeline_quant_config,
                                                 torch_dtype=dtype).to("cuda")

    if lora_name is not None:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        generator.set_adapters(lora_list)

    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"FLUX INPAINT GENERATE ERROR: {e}")

    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images


async def generate_flux_kontext(prompt,
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
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 1
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 3.5
    kwargs["image"] = image
    if seed is not None:
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in
                     range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    print("loading FluxKontextPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                                    quantization_config=pipeline_quant_config,
                                                    torch_dtype=dtype).to("cuda")

    if ip_adapter_image is not None:
        try:
            generator.load_ip_adapter("XLabs-AI/flux-ip-adapter",
                                      weight_name="ip_adapter.safetensors",
                                      image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
                                      torch_dtype=dtype)
            generator.set_ip_adapter_scale(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"FLUX KONTEXT IP ADAPTER ERROR: {e}")

    if lora_name is not None:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/flux/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX KONTEXT LORA ERROR: {e}")
        generator.set_adapters(lora_list)
        generator.fuse_lora(adapter_names=lora_list)
        generator.unload_lora_weights()

    try:
        images = generator(**kwargs).images
    except Exception as e:
        print(f"FLUX KONTEXT GENERATE ERROR: {e}")

    del generator
    torch.cuda.empty_cache()
    gc.collect()
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