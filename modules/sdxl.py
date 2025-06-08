import gc
import os
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                       DPMSolverMultistepScheduler, DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
                       DEISMultistepScheduler, UniPCMultistepScheduler, DPMSolverSDEScheduler, PNDMScheduler)
import torch
from modules.controlnet import get_sdxl_controlnet

async def generate_sdxl(prompt,
                        width,
                        height,
                        steps,
                        batch_size,
                        image=None,
                        strength=None,
                        negative_prompt=None,
                        model_name=None,
                        lora_name=None,
                        controlnet_processor=None,
                        controlnet_image=None,
                        controlnet_conditioning=None,
                        ip_adapter_image=None,
                        ip_adapter_strength=None,
                        guidance_scale=None,
                        scheduler=None,
                        seed=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 5.0
    strength = strength if strength is not None else 0.7
    controlnet_conditioning = controlnet_conditioning if controlnet_conditioning is not None else 0.5
    ip_adapter_strength = ip_adapter_strength if ip_adapter_strength is not None else 0.6
    if model_name is None:
        model_name = "misri/zavychromaxl_v100"
    if seed is not None:
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    generator, processed_image = await get_pipeline(model_name, image, controlnet_image, controlnet_processor)

    if scheduler is not None:
        match scheduler:
            case "DDIMScheduler":
                generator.scheduler = DDIMScheduler.from_config(generator.scheduler.config)
            case "DDPMScheduler":
                generator.scheduler = DDPMScheduler.from_config(generator.scheduler.config)
            case "PNDMScheduler":
                generator.scheduler = PNDMScheduler.from_config(generator.scheduler.config)
            case "LMSDiscreteScheduler":
                generator.scheduler = LMSDiscreteScheduler.from_config(generator.scheduler.config)
            case "EulerDiscreteScheduler":
                generator.scheduler = EulerDiscreteScheduler.from_config(generator.scheduler.config)
            case "HeunDiscreteScheduler":
                generator.scheduler = HeunDiscreteScheduler.from_config(generator.scheduler.config)
            case "EulerAncestralDiscreteScheduler":
                generator.scheduler = EulerAncestralDiscreteScheduler.from_config(generator.scheduler.config)
            case "DPMSolverMultistepScheduler":
                generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
            case "DPMSolverSinglestepScheduler":
                generator.scheduler = DPMSolverSinglestepScheduler.from_config(generator.scheduler.config)
            case "KDPM2DiscreteScheduler":
                generator.scheduler = KDPM2DiscreteScheduler.from_config(generator.scheduler.config)
            case "KDPM2AncestralDiscreteScheduler":
                generator.scheduler = KDPM2AncestralDiscreteScheduler.from_config(generator.scheduler.config)
            case "DEISMultistepScheduler":
                generator.scheduler = DEISMultistepScheduler.from_config(generator.scheduler.config)
            case "UniPCMultistepScheduler":
                generator.scheduler = UniPCMultistepScheduler.from_config(generator.scheduler.config)
            case "DPMSolverSDEScheduler":
                generator.scheduler = DPMSolverSDEScheduler.from_config(generator.scheduler.config)

    if ip_adapter_image is not None:
        try:
            generator.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            generator.set_ip_adapter_scale(ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"SDXL IP ADAPTER ERROR: {e}")

    if lora_name is not None:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/sdxl/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"SDXL LORA ERROR: {e}")
        generator.set_adapters(lora_list)
        generator.fuse_lora(adapter_names=lora_list)
        generator.unload_lora_weights()

    if image is not None:
        kwargs["image"] = image
        kwargs["strength"] = strength
        if controlnet_processor is not None:
            kwargs["control_image"] = processed_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning
        images = generator(**kwargs).images
    else:
        if controlnet_processor is not None:
            kwargs["image"] = processed_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning
        images = generator(**kwargs).images
    generator.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

async def get_pipeline(model_name, image, controlnet_image, controlnet_processor):
    processed_image = None
    if image is not None:
        if controlnet_image is not None:
            controlnet, processed_image = await get_sdxl_controlnet(controlnet_processor, controlnet_image)
            generator = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(model_name,
                                                                                   use_safetensors=True,
                                                                                   torch_dtype=torch.float16,
                                                                                   controlnet=controlnet,
                                                                                   ).to("cuda")
            generator.enable_vae_slicing()
        else:
            generator = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name,
                                                                         torch_dtype=torch.float16,
                                                                         use_safetensors=True).to("cuda")
            generator.enable_vae_slicing()
    else:
        if controlnet_image is not None:
            controlnet, processed_image = await get_sdxl_controlnet(controlnet_processor, controlnet_image)
            generator = StableDiffusionXLControlNetPipeline.from_pretrained(model_name,
                                                                            use_safetensors=True,
                                                                            torch_dtype=torch.float16,
                                                                            controlnet=controlnet,
                                                                            ).to("cuda")
            generator.enable_vae_slicing()
        else:
            generator = StableDiffusionXLPipeline.from_pretrained(model_name,
                                                                  torch_dtype=torch.float16,
                                                                  use_safetensors=True).to("cuda")
            generator.enable_vae_slicing()
    return generator, processed_image

async def generate_sdxl_inpaint(prompt,
                                negative_prompt,
                                width,
                                height,
                                steps,
                                batch_size,
                                image,
                                mask_image,
                                model_name,
                                lora_name=None,
                                strength=None,
                                guidance_scale=None,
                                scheduler=None,
                                seed=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["strength"] = strength if strength is not None else 0.75
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 5.0
    kwargs["image"] = image
    kwargs["mask_image"] = mask_image
    kwargs["padding_mask_crop"] = 32
    if seed is not None:
        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]
        kwargs["generator"] = generator

    if model_name is None:
        model_name = "misri/zavychromaxl_v100"

    generator = StableDiffusionXLInpaintPipeline.from_pretrained(model_name,
                                                                 torch_dtype=torch.float16,
                                                                 use_safetensors=True).to("cuda")
    generator.enable_vae_slicing()

    if scheduler is not None:
        match scheduler:
            case "DDIMScheduler":
                generator.scheduler = DDIMScheduler.from_config(generator.scheduler.config)
            case "DDPMScheduler":
                generator.scheduler = DDPMScheduler.from_config(generator.scheduler.config)
            case "PNDMScheduler":
                generator.scheduler = PNDMScheduler.from_config(generator.scheduler.config)
            case "LMSDiscreteScheduler":
                generator.scheduler = LMSDiscreteScheduler.from_config(generator.scheduler.config)
            case "EulerDiscreteScheduler":
                generator.scheduler = EulerDiscreteScheduler.from_config(generator.scheduler.config)
            case "HeunDiscreteScheduler":
                generator.scheduler = HeunDiscreteScheduler.from_config(generator.scheduler.config)
            case "EulerAncestralDiscreteScheduler":
                generator.scheduler = EulerAncestralDiscreteScheduler.from_config(generator.scheduler.config)
            case "DPMSolverMultistepScheduler":
                generator.scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
            case "DPMSolverSinglestepScheduler":
                generator.scheduler = DPMSolverSinglestepScheduler.from_config(generator.scheduler.config)
            case "KDPM2DiscreteScheduler":
                generator.scheduler = KDPM2DiscreteScheduler.from_config(generator.scheduler.config)
            case "KDPM2AncestralDiscreteScheduler":
                generator.scheduler = KDPM2AncestralDiscreteScheduler.from_config(generator.scheduler.config)
            case "DEISMultistepScheduler":
                generator.scheduler = DEISMultistepScheduler.from_config(generator.scheduler.config)
            case "UniPCMultistepScheduler":
                generator.scheduler = UniPCMultistepScheduler.from_config(generator.scheduler.config)
            case "DPMSolverSDEScheduler":
                generator.scheduler = DPMSolverSDEScheduler.from_config(generator.scheduler.config)

    if lora_name is not None:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                generator.load_lora_weights(f"loras/sdxl/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"SDXL LORA ERROR: {e}")
        generator.set_adapters(lora_list)
        generator.fuse_lora(adapter_names=lora_list)
        generator.unload_lora_weights()

    images = generator(**kwargs).images

    generator.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images
