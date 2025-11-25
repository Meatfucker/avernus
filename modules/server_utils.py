import asyncio
from io import BytesIO
import os
from typing import Optional
import subprocess

from fastapi.responses import StreamingResponse
import httpx
from huggingface_hub import snapshot_download
from loguru import logger

class ModelManager:
    def __init__(self, model_path="./models"):
        self.model_index = {}
        self.build_model_index(model_path)

    def add_model(self, arch, repo):
        snapshot_download(repo_id=repo, local_dir=f"./models/{arch}/")

    def build_model_index(self, base_path="./models"):
        model_index = {}

        # Iterate over items directly inside ./models
        for name in os.listdir(base_path):
            full_path = os.path.join(base_path, name)

            # Only consider directories as keys
            if os.path.isdir(full_path):
                # List only immediate files/directories inside each model directory
                contents = os.listdir(full_path)
                model_index[name] = contents
        self.model_index = model_index

    def get_index(self):
        return self.model_index

class ServerManager:
    def __init__(self,
                 current_process: Optional[subprocess.Popen] = None,
                 model_type: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.current_process = current_process
        self.model_type = model_type
        self.model_name = model_name

    async def set_pipeline(self, model_type=None, model_name=None):
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

