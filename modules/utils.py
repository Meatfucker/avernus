import asyncio
import base64
from io import BytesIO
import math
import os
import psutil
from typing import Optional
import subprocess

from fastapi.responses import StreamingResponse
import httpx
from loguru import logger
import torch
from PIL import Image
import pynvml


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
                logger.info(f"Killing {self.model_type}, {self.model_name} server ({self.current_process.pid})")
                self.current_process.kill()
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

def image_to_base64(image):
    """Takes a PIL image and converts it to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_memory():
    ram = psutil.virtual_memory()
    pynvml.nvmlInit()
    device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    vram_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2  # MB
    pynvml.nvmlShutdown()
    current_memory = f"VRAM Allocated:{vram_used:.2f}MB / RAM Allocated:{ram.used / 1024**2:.2f}MB"
    return current_memory

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
            return {"error": "Failed to reach image generation server"}
        except httpx.HTTPStatusError as exc:
            logger.error(f"Bad response from image generation server: {exc}")
            return {"error": f"Image generation failed with status {exc.response.status_code}"}

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
            return {"error": "Failed to reach ACE generation server"}
        except httpx.HTTPStatusError as exc:
            logger.error(f"Bad response from ACE generation server: {exc}")
            return {"error": f"ACE generation failed with status {exc.response.status_code}"}
