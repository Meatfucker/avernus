import gc
import os
import torch
from diffusers import LTXPipeline
from diffusers.hooks import apply_group_offloading

directory = "/mnt/1000gb/ml/offload"

async def generate_ltx(prompt,
                       input_video=None,
                       height=512,
                       width=704,
                       num_frames=161,
                       guidance_scale=3,
                       seed=None):

    generator = LTXPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
    generator = await apply_offloading(generator)
    
    video = generator(prompt=prompt,
                      width=width,
                      height=height,
                      num_frames=num_frames,
                      num_inference_steps=50).frames[0]



    del generator
    gc.collect()
    torch.cuda.empty_cache()
    await clear_offload_directory()
    return video

async def apply_offloading(generator):
    generator.transformer.enable_group_offload(onload_device="cuda",
                                               offload_device="cpu",
                                               offload_type="block_level",
                                               num_blocks_per_group=1,
                                               offload_to_disk_path=directory,
                                               use_stream=True,
                                               record_stream=True,
                                               non_blocking=False)
    apply_group_offloading(generator.text_encoder,
                           onload_device="cuda",
                           offload_device="cpu",
                           offload_type="block_level",
                           num_blocks_per_group=1,
                           offload_to_disk_path=directory,
                           use_stream=True,
                           record_stream=True,
                           non_blocking=False)
    for name, component in generator.components.items():
        if name not in ["transformer", "text_encoder"] and isinstance(component, torch.nn.Module):
            component.cuda()
    return generator


async def clear_offload_directory():
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
