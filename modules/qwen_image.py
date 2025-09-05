from diffusers import QwenImagePipeline, QwenImageEditPipeline, QwenImageInpaintPipeline, QwenImageImg2ImgPipeline, QwenImageTransformer2DModel
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import TorchAoConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
import math
import os

dtype = torch.bfloat16

async def load_qwen_image_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "qwen_image":
        print("loading QwenImagePipeline")
        await avernus_pipeline.delete_pipeline()
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod",
                                   "time_text_embed",
                                   "img_in",
                                   "norm_out",
                                   "proj_out",
                                   "img_mod",
                                   "txt_mod",
                                   ],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        transformer = transformer.to("cpu")
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="text_encoder",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.to("cpu")
        generator = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image",
                                                      transformer=transformer,
                                                      text_encoder=text_encoder,
                                                      torch_dtype=dtype)
        generator.enable_model_cpu_offload()
        generator.enable_vae_slicing()
        await avernus_pipeline.set_pipeline(generator, "qwen_image")
    return avernus_pipeline


async def load_qwen_image_inpaint_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "qwen_image_inpaint":
        print("loading QwenImageInpaintPipeline")
        await avernus_pipeline.delete_pipeline()
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod",
                                   "time_text_embed",
                                   "img_in",
                                   "norm_out",
                                   "proj_out",
                                   "img_mod",
                                   "txt_mod",
                                   ],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        transformer = transformer.to("cpu")
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="text_encoder",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.to("cpu")
        generator = QwenImageInpaintPipeline.from_pretrained("Qwen/Qwen-Image",
                                                             transformer=transformer,
                                                             text_encoder=text_encoder,
                                                             torch_dtype=dtype)
        generator.enable_model_cpu_offload()
        generator.enable_vae_slicing()
        await avernus_pipeline.set_pipeline(generator, "qwen_image_inpaint")

    return avernus_pipeline


async def load_qwen_image2image_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "qwen_image2image":
        print("loading QwenImage2ImagePipeline")
        await avernus_pipeline.delete_pipeline()
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod",
                                   "time_text_embed",
                                   "img_in",
                                   "norm_out",
                                   "proj_out",
                                   "img_mod",
                                   "txt_mod",
                                   ],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        transformer = transformer.to("cpu")
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Image",
            subfolder="text_encoder",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.to("cpu")
        generator = QwenImageImg2ImgPipeline.from_pretrained("Qwen/Qwen-Image",
                                                             transformer=transformer,
                                                             text_encoder=text_encoder,
                                                             torch_dtype=dtype)
        generator.enable_model_cpu_offload()
        generator.enable_vae_slicing()
        await avernus_pipeline.set_pipeline(generator, "qwen_image2image")
    return avernus_pipeline


async def load_qwen_image_edit_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "qwen_image_edit":
        print("loading QwenImageEditPipeline")
        await avernus_pipeline.delete_pipeline()
        quantization_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod",
                                   "time_text_embed",
                                   "img_in",
                                   "norm_out",
                                   "proj_out",
                                   "img_mod",
                                   "txt_mod",
                                   ],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        transformer = transformer.to("cpu")
        quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            subfolder="text_encoder",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.to("cpu")
        generator = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit",
                                                          transformer=transformer,
                                                          text_encoder=text_encoder,
                                                          torch_dtype=dtype)
        generator.enable_model_cpu_offload()
        generator.enable_vae_slicing()
        await avernus_pipeline.set_pipeline(generator, "qwen_image_edit")
    return avernus_pipeline

async def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator


async def load_qwen_loras(avernus_pipeline, lora_name):
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0]
                avernus_pipeline.pipeline.load_lora_weights(f"loras/qwen/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"QWEN LORA ERROR: {e}")
        avernus_pipeline.pipeline.set_adapters(lora_list)
        return avernus_pipeline
    except Exception as e:
        print(f"QWEN LORA ERROR: {e}")


async def generate_qwen_image(avernus_pipeline,
                              prompt,
                              width,
                              height,
                              steps,
                              batch_size,
                              negative_prompt=None,
                              image=None,
                              strength=None,
                              lora_name=None,
                              true_cfg_scale=None,
                              seed=None):

    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    strength = strength if strength is not None else 0.7
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 4.0
    if seed is not None:
        kwargs["generator"] = await get_seed_generators(kwargs["num_images_per_prompt"], seed)

    if image is not None:
        kwargs["image"] = image
        kwargs["strength"] = strength
        avernus_pipeline = await load_qwen_image2image_pipeline(avernus_pipeline)
    else:
        avernus_pipeline = await load_qwen_image_pipeline(avernus_pipeline)

    if lora_name is not None:
        avernus_pipeline = await load_qwen_loras(avernus_pipeline, lora_name)

    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()

    except Exception as e:
        print(f"QWEN IMAGE GENERATE ERROR: {e}")

    return images


async def generate_qwen_image_inpaint(avernus_pipeline,
                                      prompt,
                                      width,
                                      height,
                                      steps,
                                      batch_size,
                                      negative_prompt=None,
                                      image=None,
                                      mask_image=None,
                                      strength=None,
                                      lora_name=None,
                                      true_cfg_scale=None,
                                      seed=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["width"] = width if width is not None else 1024
    kwargs["height"] = height if height is not None else 1024
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 4
    kwargs["strength"] = strength if strength is not None else 0.7
    kwargs["image"] = image
    kwargs["mask_image"] = mask_image
    kwargs["padding_mask_crop"] = 32
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 4.0

    if seed is not None:
        kwargs["generator"] = await get_seed_generators(kwargs["num_images_per_prompt"], seed)

    avernus_pipeline = await load_qwen_image_inpaint_pipeline(avernus_pipeline)

    if lora_name is not None:
        avernus_pipeline = await load_qwen_loras(avernus_pipeline, lora_name)

    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
    except Exception as e:
        print(f"QWEN IMAGE INPAINT GENERATE ERROR: {e}")

    return images


async def generate_qwen_image_edit(avernus_pipeline,
                                   prompt,
                                   width,
                                   height,
                                   steps,
                                   batch_size,
                                   image,
                                   negative_prompt=None,
                                   lora_name=None,
                                   true_cfg_scale=None,
                                   seed=None):
    qwen_image_width, qwen_image_height = resize_by_pixels(image.width, image.height)
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["num_inference_steps"] = steps if steps is not None else 30
    kwargs["num_images_per_prompt"] = batch_size if batch_size is not None else 1
    kwargs["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 4.0
    kwargs["image"] = image
    if width is not None:
        kwargs["width"] = width
    else:
        kwargs["width"] = qwen_image_width
    if height is not None:
        kwargs["height"] = height
    else:
        kwargs["height"] = qwen_image_height
    if seed is not None:
        kwargs["generator"] = await get_seed_generators(kwargs["num_images_per_prompt"], seed)

    avernus_pipeline = await load_qwen_image_edit_pipeline(avernus_pipeline)

    if lora_name is not None:
        avernus_pipeline = await load_qwen_loras(avernus_pipeline, lora_name)

    try:
        images = avernus_pipeline.pipeline(**kwargs).images
        if lora_name is not None:
            avernus_pipeline.pipeline.unload_lora_weights()
    except Exception as e:
        print(f"QWEN IMAGE EDIT GENERATE ERROR: {e}")

    return images


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