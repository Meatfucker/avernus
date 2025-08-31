import gc
import os
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLControlNetImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                       DPMSolverMultistepScheduler, DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
                       DEISMultistepScheduler, UniPCMultistepScheduler, DPMSolverSDEScheduler, PNDMScheduler)
from diffusers.quantizers import PipelineQuantizationConfig
import torch
from modules.controlnet import get_sdxl_controlnet, get_controlnet_image

async def generate_sdxl(avernus_pipeline,
                        prompt,
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
        kwargs["generator"] = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(kwargs["num_images_per_prompt"])]

    avernus_pipeline, processed_image = await get_pipeline(avernus_pipeline, model_name, image, controlnet_image, controlnet_processor)

    if scheduler is not None:
        await set_scheduler(avernus_pipeline, scheduler)

    if ip_adapter_image is not None:
        try:
            avernus_pipeline = await load_ip_adapters(avernus_pipeline, ip_adapter_strength)
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception as e:
            print(f"SDXL IP ADAPTER ERROR: {e}")

    if lora_name is not None:
        avernus_pipeline = await load_sdxl_loras(avernus_pipeline, lora_name)

    avernus_pipeline.pipeline.enable_vae_slicing()

    if image is not None:
        kwargs["image"] = image
        kwargs["strength"] = strength
        if controlnet_processor is not None:
            kwargs["control_image"] = processed_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
        if ip_adapter_image is not None:
            avernus_pipeline.pipeline.unload_ip_adapter()
    else:
        if controlnet_processor is not None:
            kwargs["image"] = processed_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
        if ip_adapter_image is not None:
            avernus_pipeline.pipeline.unload_ip_adapter()

    return images

async def get_pipeline(avernus_pipeline, model_name, image, controlnet_image, controlnet_processor):
    processed_image = None
    if image is not None:
        if controlnet_image is not None:
            processed_image = await get_controlnet_image(controlnet_processor, controlnet_image)
            if avernus_pipeline.model_type != "sdxl_i2i_controlnet" or avernus_pipeline.model_name != model_name:
                print("Loading SdxlI2IControlnetPipeline")
                await avernus_pipeline.delete_pipeline()
                controlnet = await get_sdxl_controlnet(controlnet_processor)
                generator = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(model_name,
                                                                                       use_safetensors=True,
                                                                                       torch_dtype=torch.bfloat16,
                                                                                       controlnet=controlnet,
                                                                                       ).to("cuda")
                await avernus_pipeline.set_pipeline(generator, "sdxl_i2i_controlnet", model_name)
        else:
            if avernus_pipeline.model_type != "sdxl_i2i" or avernus_pipeline.model_name != model_name:
                print("Loading SdxlI2IPipeline")
                await avernus_pipeline.delete_pipeline()
                generator = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name,
                                                                             torch_dtype=torch.bfloat16,
                                                                             use_safetensors=True).to("cuda")
                await avernus_pipeline.set_pipeline(generator, "sdxl_i2i", model_name)
    else:
        if controlnet_image is not None:
            processed_image = await get_controlnet_image(controlnet_processor, controlnet_image)
            if avernus_pipeline.model_type != "sdxl_controlnet" or avernus_pipeline.model_name != model_name:
                print("Loading SdxlControlnetPipeline")
                await avernus_pipeline.delete_pipeline()
                controlnet = await get_sdxl_controlnet(controlnet_processor)
                generator = StableDiffusionXLControlNetPipeline.from_pretrained(model_name,
                                                                                use_safetensors=True,
                                                                                torch_dtype=torch.bfloat16,
                                                                                controlnet=controlnet,
                                                                                ).to("cuda")
                await avernus_pipeline.set_pipeline(generator, "sdxl_controlnet", model_name)
        else:
            if avernus_pipeline.model_type != "sdxl" or avernus_pipeline.model_name != model_name:
                print("Loading SdxlPipeline")
                await avernus_pipeline.delete_pipeline()
                generator = StableDiffusionXLPipeline.from_pretrained(model_name,
                                                                      torch_dtype=torch.bfloat16,
                                                                      use_safetensors=True).to("cuda")
                await avernus_pipeline.set_pipeline(generator, "sdxl", model_name)

    return avernus_pipeline, processed_image

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
                                                                 torch_dtype=torch.bfloat16,
                                                                 use_safetensors=True).to("cuda")


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

    images = generator(**kwargs).images



    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return images

async def set_scheduler(avernus_pipeline, scheduler):
    try:
        match scheduler:
            case "DDIMScheduler":
                avernus_pipeline.pipeline.scheduler = DDIMScheduler.from_config(avernus_pipeline.pipeline.scheduler.config)
            case "DDPMScheduler":
                avernus_pipeline.pipeline.scheduler = DDPMScheduler.from_config(avernus_pipeline.pipeline.scheduler.config)
            case "PNDMScheduler":
                avernus_pipeline.pipeline.scheduler = PNDMScheduler.from_config(avernus_pipeline.pipeline.scheduler.config)
            case "LMSDiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = LMSDiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "EulerDiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "HeunDiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = HeunDiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "EulerAncestralDiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "DPMSolverMultistepScheduler":
                avernus_pipeline.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "DPMSolverSinglestepScheduler":
                avernus_pipeline.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "KDPM2DiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = KDPM2DiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "KDPM2AncestralDiscreteScheduler":
                avernus_pipeline.pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "DEISMultistepScheduler":
                avernus_pipeline.pipeline.scheduler = DEISMultistepScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "UniPCMultistepScheduler":
                avernus_pipeline.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
            case "DPMSolverSDEScheduler":
                avernus_pipeline.pipeline.scheduler = DPMSolverSDEScheduler.from_config(
                    avernus_pipeline.pipeline.scheduler.config)
    except Exception as e:
        print(f"SCHEDULER ERROR: {e}")

async def load_ip_adapters(avernus_pipeline, strength):
    try:
        avernus_pipeline.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin",
                                                  device="cuda")
        avernus_pipeline.pipeline.set_ip_adapter_scale(strength)
        return avernus_pipeline
    except Exception as e:
        print(f"SDXL IP ADAPTER ERROR: {e}")

async def load_sdxl_loras(avernus_pipeline, lora_name):
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                avernus_pipeline.pipeline.load_lora_weights(f"loras/sdxl/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"SDXL LORA ERROR: {e}")
        avernus_pipeline.pipeline.set_adapters(lora_list)
        return avernus_pipeline
    except Exception as e:
        print(f"SDXL LORA ERROR: {e}")