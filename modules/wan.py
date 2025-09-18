import math
import torch
from diffusers import AutoModel, WanPipeline, WanImageToVideoPipeline
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import UMT5EncoderModel


async def load_wan_components(model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers"):
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["time_embedder", "timesteps_proj", "time_proj", "norm_out", "proj_out"],
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(model_name,
                                            subfolder="transformer",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=transformer_quantization_config
                                            )

    return text_encoder, vae, transformer

async def load_wan_t2v_pipeline(avernus_pipeline, model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers"):
    if avernus_pipeline.model_type != "wan-t2v":
        print("loading WanT2VPipeline")
        await avernus_pipeline.delete_pipeline()
        text_encoder, vae, transformer = await load_wan_components()

        pipeline = WanPipeline.from_pretrained(
            model_name,
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()
        await avernus_pipeline.set_pipeline(pipeline, "wan-t2v")
    return avernus_pipeline

async def load_wan_i2v_pipeline(avernus_pipeline, model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers"):
    if avernus_pipeline.model_type != "wan-i2v":
        print("loading WanI2VPipeline")
        await avernus_pipeline.delete_pipeline()
        text_encoder, vae, transformer = await load_wan_components()
        pipeline = WanImageToVideoPipeline.from_pretrained(
            model_name,
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()
        await avernus_pipeline.set_pipeline(pipeline, "wan-i2v")
    return avernus_pipeline

async def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

async def generate_wan_ti2v(avernus_pipeline,
                            prompt: str,
                            image: str = None,
                            negative_prompt: str = None,
                            num_frames: int = 81,
                            guidance_scale: float = 5.0,
                            height: int = None,
                            width: int = None,
                            seed: int = None,
                            model_name: str = None):
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["negative_prompt"] = negative_prompt if negative_prompt is not None else ""
    kwargs["num_frames"] = num_frames
    kwargs["guidance_scale"] = guidance_scale
    kwargs["height"] = height
    kwargs["width"] = width
    if model_name is None:
        model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    if seed is not None:
        kwargs["generator"] = await get_seed_generators(1, seed)
    if image is not None:
        image_width, image_height = resize_by_pixels(image.width, image.height)

        if width is not None:
            kwargs["width"] = width
        else:
            kwargs["width"] = image_width
        if height is not None:
            kwargs["height"] = height
        else:
            kwargs["height"] = image_height
        kwargs["image"] = image

        print(f"W: {kwargs['width']} H:{kwargs['height']}")
        avernus_pipeline = await load_wan_i2v_pipeline(avernus_pipeline, model_name)
    else:
        if width is not None:
            kwargs["width"] = width
        else:
            kwargs["width"] = 832

        if height is not None:
            kwargs["height"] = height
        else:
            kwargs["height"] = 480
        avernus_pipeline = await load_wan_t2v_pipeline(avernus_pipeline, model_name)
    output = avernus_pipeline.pipeline(**kwargs).frames[0]

    return output

def resize_by_pixels(width, height, target_pixels=832*480, keep_if_within=0.0):
    """
    Return (new_width, new_height) so total pixels ~= target_pixels,
    preserving aspect ratio. Ensures both dimensions are divisible by 16.
    If current pixels are within ±keep_if_within (e.g. 0.25 for 25%),
    the original size is returned (but adjusted to multiple of 16).
    """
    def round_to_multiple(value, base=16):
        return max(base, int(round(value / base) * base))

    current = width * height
    if keep_if_within > 0 and abs(current - target_pixels) / target_pixels <= keep_if_within:
        return round_to_multiple(width), round_to_multiple(height)

    scale = math.sqrt(target_pixels / current)
    new_w = round_to_multiple(width * scale)
    new_h = round_to_multiple(height * scale)
    print(f"W: {new_w} H:{new_h}")
    return new_w, new_h