import gc
import os
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.quantizers.quantization_config import QuantoConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig

async def generate_wan(prompt: str,
                       negative_prompt: str = None,
                       num_frames: int = 81,
                       guidance_scale: float = 5.0,
                       height: int = 480,
                       width: int = 832,
                       seed: int = None,
                       input_video = None):
#    pipeline_quant_config = PipelineQuantizationConfig(
#        quant_backend="bitsandbytes_4bit",
#        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
#        components_to_quantize=["transformer", "text_encoder", "vae"],
#    )
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_mapping={
            "transformer": QuantoConfig(weights_dtype="float8"),
            "text_encoder": TransformersBitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16),
        }
    )

    pipeline = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        torch_dtype=torch.bfloat16,
        quantization_config=pipeline_quant_config
    ).to("cuda")
    # group-offloading
    #onload_device = torch.device("cuda")
    #offload_device = torch.device("cpu")
    #apply_group_offloading(pipeline.text_encoder,
    #                       onload_device=onload_device,
    #                       offload_device=offload_device,
    #                       offload_type="block_level",
    #                       num_blocks_per_group=4
    #                       )
    #pipeline.transformer.enable_group_offload(
    #    onload_device=onload_device,
    #    offload_device=offload_device,
    #    offload_type="leaf_level",
    #    use_stream=True
    #)
    pipeline.enable_model_cpu_offload()

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    ).frames[0]
    return output