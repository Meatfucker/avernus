import asyncio
import base64
from io import BytesIO
import logging
import math
import os
import psutil
import sys
from typing import Optional
import subprocess

from fastapi.responses import StreamingResponse, FileResponse
import httpx
from loguru import logger
import torch
from PIL import Image
import pynvml


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Don't log requests to `/ping`
        return "/status" not in record.getMessage()

def setup_logging():
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]


def base64_to_image(base64_string):
    """Takes a base64 image and converts it to a PIL image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def get_memory():
    ram = psutil.virtual_memory()
    pynvml.nvmlInit()
    device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    vram_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2  # MB
    pynvml.nvmlShutdown()
    current_memory = f"VRAM Allocated:{vram_used:.2f}MB / RAM Allocated:{ram.used / 1024**2:.2f}MB"
    return current_memory

def get_seed_generators(amount, seed):
    generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(amount)]
    return generator

def image_to_base64(image):
    """Takes a PIL image and converts it to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_loras(pipeline, arch, lora_name):
    try:
        lora_list = []
        for lora in lora_name:
            try:
                lora_name = os.path.splitext(lora)[0].strip('.')
                pipeline.load_lora_weights(f"./loras/{arch}/{lora}", adapter_name=lora_name)
                lora_list.append(lora_name)
            except Exception as e:
                print(f"FLUX LORA ERROR: {e}")
        pipeline.set_adapters(lora_list)
    except Exception as e:
        logger.error(f"{arch} LORA LOAD ERROR:{e}")
        pass
    return pipeline

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

def return_loras(path):
    """Returns a list of available loras in the supplied directory"""
    try:
        if not os.path.exists(path):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return filenames
    except Exception as e:
        return e

