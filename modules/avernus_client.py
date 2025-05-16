import httpx

class AvernusClient:
    """This is the client for the avernus API server"""
    def __init__(self, url, port=6969):
        self.url = url
        self.port = port
        self.base_url = f"{self.url}:{self.port}"

    async def rag_retrieve(self, prompt, max_candidates=20, similarity_threshold=0.6):
        """This takes a prompt and optionally a number of results, and then returns the mathing RAG documents"""
        url = f"http://{self.base_url}/rag_retrieve"
        data = {"prompt": prompt, "max_candidates": max_candidates, "similarity_threshold": similarity_threshold}

        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"RAG ERROR: {response.status_code}, Response: {response.text}")
                return {"RAG ERROR": response.text}
        except Exception as e:
            print(f"EXCEPTION ERROR: {e}")
            return {"ERROR": str(e)}

    async def llm_chat(self, prompt, model_name=None, messages=None):
        """This takes a prompt, and optionally a model name and chat history, then returns a response"""
        url = f"http://{self.base_url}/llm_chat"
        data = {"prompt": prompt, "model_name": model_name, "messages": messages}

        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"LLM ERROR: {response.status_code}, Response: {response.text}")
                return {"LLM ERROR": response.text}
        except Exception as e:
            print(f"EXCEPTION ERROR: {e}")
            return {"ERROR": str(e)}

    async def multimodal_llm_chat(self, prompt, model_name=None, messages=None):
        """This takes a prompt, and optionally a model name and chat history, then returns a response"""
        url = f"http://{self.base_url}/multimodal_llm_chat"
        data = {"prompt": prompt, "model_name": model_name, "messages": messages}

        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"MULTIMODAL LLM ERROR: {response.status_code}, Response: {response.text}")
                return {"MULTIMODAL LLM ERROR": response.text}
        except Exception as e:
            print(f"EXCEPTION ERROR: {e}")
            return {"ERROR": str(e)}

    async def sdxl_image(self, prompt, image=None, negative_prompt=None, model_name=None, lora_name=None, width=None,
                         height=None, steps=None, batch_size=None, strength=None, controlnet_image=None,
                         controlnet_processor=None, controlnet_conditioning=None, ip_adapter_image=None,
                         ip_adapter_strength=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sdxl_generate"
        data = {"prompt": prompt,
                "image": image,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "controlnet_image": controlnet_image,
                "controlnet_processor": controlnet_processor,
                "controlnet_conditioning": controlnet_conditioning,
                "ip_adapter_strength": ip_adapter_strength,
                "ip_adapter_image": ip_adapter_image}
        try:
            async with httpx.AsyncClient(timeout=3600) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json().get("images", [])
            else:
                print(f"SDXL ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_image(self, prompt, image=None, model_name=None, lora_name=None, width=None, height=None, steps=None,
                         batch_size=None, strength=None, controlnet_image=None,
                         controlnet_processor=None, ip_adapter_image=None,
                         ip_adapter_strength=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_generate"
        data = {"prompt": prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "controlnet_image": controlnet_image,
                "controlnet_processor": controlnet_processor,
                "ip_adapter_strength": ip_adapter_strength,
                "ip_adapter_image": ip_adapter_image}
        try:
            async with httpx.AsyncClient(timeout=3600) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json().get("images", [])
            else:
                print(f"FLUX ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_loras(self):
        """Fetches the list of sdxl LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_sdxl_loras"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json().get("loras", [])
            else:
                print(f"LIST SDXL LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sdxl_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_controlnets(self):
        """Fetches the list of sdxl controlnets from the server."""
        url = f"http://{self.base_url}/list_sdxl_controlnets"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json().get("sdxl_controlnets", [])
            else:
                print(f"LIST SDXL CONTROLNETS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sdxl_controlnets ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_flux_loras(self):
        """Fetches the list of flux LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_flux_loras"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json().get("loras", [])
            else:
                print(f"LIST FLUX LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_flux_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_flux_controlnets(self):
        """Fetches the list of flux controlnets from the server."""
        url = f"http://{self.base_url}/list_flux_controlnets"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json().get("flux_controlnets", [])
            else:
                print(f"LIST Flux CONTROLNETS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_flux_controlnets ERROR: {e}")
            return {"ERROR": str(e)}

    async def check_status(self):
        """Attempts to contact the avernus server and returns a dict with status information from the server"""
        url = f"http://{self.base_url}/status"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"STATUS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"status ERROR: {e}")
            return {"ERROR": str(e)}