import requests
import httpx
from loguru import logger

class AvernusClient:
    """This is the client for the avernus API server"""
    def __init__(self, url, port=6969):
        self.url = url
        self.port = port
        self.base_url = f"{self.url}:{self.port}"

    async def llm_chat(self, prompt, model_name=None, messages=None):
        """This takes a prompt, and optionally a model name and chat history, then returns a response"""
        url = f"http://{self.base_url}/llm_chat"
        data = {"prompt": prompt, "model_name": model_name, "messages": messages}

        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                logger.info(f"STATUS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            logger.info(f"EXCEPTION ERROR: {e}")
            return {"ERROR": str(e)}

    async def sdxl_image(self, prompt, negative_prompt=None, model_name=None, lora_name=None, width=None, height=None,
                         steps=None, batch_size=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sdxl_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size}
        try:
            async with httpx.AsyncClient(timeout=3600) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                logger.info(f"ERROR: {response.status_code}")
        except Exception as e:
            logger.info(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_image(self, prompt, negative_prompt=None, model_name=None, lora_name=None, width=None, height=None, steps=None,
                         batch_size=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size}
        try:
            async with httpx.AsyncClient(timeout=3600) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                logger.info(f"ERROR: {response.status_code}")
        except Exception as e:
            logger.info(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_loras(self):
        """Fetches the list of sdxl LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_sdxl_loras"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.info(f"STATUS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            logger.error(f"list_sdxl_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_flux_loras(self):
        """Fetches the list of flux LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_flux_loras"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.info(f"STATUS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            logger.error(f"list_flux_loras ERROR: {e}")
            return {"ERROR": str(e)}
