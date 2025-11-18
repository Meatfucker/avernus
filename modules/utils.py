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


class ServerManager:
    def __init__(self,
                 current_process: Optional[subprocess.Popen] = None,
                 model_type: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.current_process = current_process
        self.model_type = model_type
        self.model_name = model_name

    async def set_pipeline(self, model_type=None, model_name=None):
        logger.info(get_memory())
        if self.model_type != model_type or self.model_name != model_name:
            if self.current_process is not None:
                self.kill_pipeline()
            logger.info(f"Starting {model_type}, {model_name} server...")
            try:
                path = f"modules/{model_type}.py"
                self.current_process = subprocess.Popen(["python", path])
                await self.wait_until_online("http://localhost:6970/online", interval=1.0, timeout=60.0)
                self.model_type = model_type
                self.model_name = model_name
            except Exception as e:
                print(e)
        else:
            pass

    def kill_pipeline(self):
        if self.current_process is not None:
            logger.info(f"Killing {self.model_type}, {self.model_name} server ({self.current_process.pid})")
            self.current_process.kill()
            self.current_process = None
            self.model_type = None
            self.model_name = None

    def get_model_name(self):
        return self.model_name

    def get_model_type(self):
        return self.model_type

    async def wait_until_online(self, url: str, interval: float = 1.0, timeout: float = 60.0):
        """
        Poll the API until it returns True or until timeout is reached.
        """
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            logger.info("Waiting for server to come online...")
            while True:
                try:
                    response = await client.get(url, timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        # Adjust this according to your API’s response format
                        if data is True or data.get("online") is True:
                            return
                except Exception:
                    # Likely connection refused while server starts up
                    pass

                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(f"Server at {url} did not become ready within {timeout} seconds")

                await asyncio.sleep(interval)


def base64_to_image(base64_string):
    """Takes a base64 image and converts it to a PIL image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def cleanup_and_stream(tmp_path):
    with open(tmp_path, "rb") as f:
        yield from f
    os.remove(tmp_path)  # Remove temp file after streaming
    return FileResponse(cleanup_and_stream(tmp_path), media_type="video/mp4")

async def forward_post_request(url: str, data) -> dict:
    """
    Forwards a POST request with JSON data to a specified URL.

    Args:
        url (str): The endpoint to forward to.
        data: Object with a `.dict()` method (e.g., Pydantic model).

    Returns:
        dict: JSON response or an error dict.
    """
    async with httpx.AsyncClient(timeout=3600.0) as client:
        try:
            response = await client.post(url, json=data.dict())
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            logger.error(f"Error while forwarding request: {exc}")
            return {"error": "Failed to reach generation server"}
        except httpx.HTTPStatusError as exc:
            logger.error(f"Bad response from image generation server: {exc}")
            return {"error": f"generation failed with status {exc.response.status_code}"}

async def forward_stream_request(url: str, data) -> StreamingResponse:
    """
    Forwards a POST request and streams the binary response.

    Args:
        url (str): Target URL.
        data: Pydantic model or dict.

    Returns:
        StreamingResponse: Passes through the streamed response.
    """
    async with httpx.AsyncClient(timeout=3600.0) as client:
        try:
            response = await client.post(url, json=data.dict())
            response.raise_for_status()

            # Stream the audio back to the client
            return StreamingResponse(
                BytesIO(response.content),
                media_type=response.headers.get("content-type", "application/octet-stream")
            )
        except httpx.RequestError as exc:
            logger.error(f"Error while forwarding request: {exc}")
            return {"error": "Failed to reach generation server"}
        except httpx.HTTPStatusError as exc:
            logger.error(f"Bad response from ACE generation server: {exc}")
            return {"error": f"Generation failed with status {exc.response.status_code}"}

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

def print_memory(heading: str = None):
    if heading is not None:
        print(heading)
    process = psutil.Process()
    print(f"VRAM Allocated:{torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB VRAM Reserved:{torch.cuda.memory_reserved() / 1024 ** 2:.2f}MB RAM Allocated:{process.memory_info().rss / 1024 ** 2:.2f}MB")
    return

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

