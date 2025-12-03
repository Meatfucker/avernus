import aiofiles
import httpx
import os

class AvernusClient:
    """This is the client for the avernus API server"""
    def __init__(self, url, port=6969):
        self.url = url
        self.port = port
        self.base_url = f"{self.url}:{self.port}"

    async def ace_music(self, prompt, lyrics, audio_duration=None, guidance_scale=None, infer_step=None,
                        omega_scale=None, actual_seeds=None):
        """This takes a prompt and lyrics and returns a song"""
        url = f"http://{self.base_url}/ace_generate"
        data = {"prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration,
                "guidance_scale": guidance_scale,
                "infer_step": infer_step,
                "omega_scale": omega_scale,
                "actual_seeds": actual_seeds}

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "audio": response.content,
                        }

            else:
                print(f"ACE STEP ERROR: {response.status_code} - {response.text}")
                return {"status": status_header}
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def auraflow_image(self, prompt, negative_prompt=None, model_name=None, width=None, height=None, steps=None,
                            batch_size=None, seed=None, guidance_scale=None, lora_name=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/auraflow_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"AURAFLOW ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def check_status(self):
        """Attempts to contact the avernus server and returns a dict with status information from the server"""
        url = f"http://{self.base_url}/status"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"STATUS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"status ERROR: {e}")
            return {"ERROR": str(e)}

    async def chroma_image(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None, width=None,
                           height=None, steps=None, batch_size=None, strength=None, seed=None, guidance_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/chroma_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "seed": seed,
                "guidance_scale": guidance_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"CHROMA ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def chronoedit(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None,
                         width=None, height=None, steps=None, batch_size=None, seed=None, guidance_scale=None,
                         flow_shift=None, num_frames=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/chronoedit_generate"
        data = {"prompt": prompt,
                "negative_prompt":  negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "flow_shift": flow_shift,
                "num_frames": num_frames}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"CHRONOEDIT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_fill_image(self, prompt, image=None, model_name=None, width=None,
                              height=None, steps=None, batch_size=None, guidance_scale=None, mask_image=None,
                              strength=None, lora_name=None, seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_fill_generate"
        data = {"prompt": prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "mask_image": mask_image,
                "strength": strength,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FLUX FILL ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_image(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None, width=None,
                         height=None, steps=None, batch_size=None, strength=None, ip_adapter_image=None,
                         ip_adapter_strength=None, seed=None, guidance_scale=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "ip_adapter_strength": ip_adapter_strength,
                "ip_adapter_image": ip_adapter_image,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FLUX ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_inpaint_image(self, prompt, negative_prompt=None, image=None, model_name=None, width=None,
                                 height=None, steps=None, batch_size=None, guidance_scale=None, mask_image=None,
                                 strength=None, lora_name=None, seed=None, true_cfg_scale=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_inpaint_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "mask_image": mask_image,
                "strength": strength,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FLUX INPAINT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux_kontext(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None, width=None,
                           height=None, steps=None, batch_size=None, controlnet_image=None, controlnet_processor=None,
                           ip_adapter_image=None, ip_adapter_strength=None, seed=None, guidance_scale=None,
                           true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux_kontext_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "controlnet_image": controlnet_image,
                "controlnet_processor": controlnet_processor,
                "ip_adapter_strength": ip_adapter_strength,
                "ip_adapter_image": ip_adapter_image,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FLUX KONTEXT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def flux2_image(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None, width=None,
                         height=None, steps=None, batch_size=None, seed=None, guidance_scale=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/flux2_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FLUX2 ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def framepack(self, prompt, image, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, last_image=None, seed=None, model_name=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/framepack_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "image": image,
                "last_image": last_image,
                "model_name": model_name,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"FRAMEPACK ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def hidream_image(self, prompt, negative_prompt=None, model_name=None, width=None, height=None, steps=None,
                            batch_size=None, seed=None, guidance_scale=None, lora_name=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/hidream_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HIDREAM ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def hunyuan_ti2v(self, prompt, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, image=None,  seed=None, model_name=None, flow_shift=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/hunyuan_ti2v_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "image": image,
                "model_name": model_name,
                "flow_shift": flow_shift,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"HUNYUAN TI2V ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def image_gen_aux_upscale(self, image, model=None, scale=None, tiling=None, tile_width=None, tile_height=None, overlap=None):
        """This takes an image and an optional scale of either 2, 4, or 8 and returns an upscaled image"""
        url = f"http://{self.base_url}/image_gen_aux_upscale"
        data = {"image": image,
                "model": model,
                "scale": scale,
                "tiling": tiling,
                "tile_width": tile_width,
                "tile_height": tile_height,
                "overlap": overlap}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"IMAGE_GEN_AUX_UPSCALE ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def kandinsky5_t2v(self, prompt, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, seed=None, model_name=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/kandinsky5_t2v_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "model_name": model_name,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"KANDINSKY5 T2V ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}


    async def list_chroma_loras(self):
        """Fetches the list of flux LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_chroma_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST CHROMA LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_chroma_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_flux_loras(self):
        """Fetches the list of flux LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_flux_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST FLUX LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_flux_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_flux2_loras(self):
        """Fetches the list of flux2 LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_flux2_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST FLUX2 LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_flux2_loras ERROR: {e}")
            return {"ERROR": str(e)}


    async def list_models(self):
        """Fetches a list of available model types and models"""
        url = f"http://{self.base_url}/list_models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"MODEL LIST ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_models ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_qwen_image_loras(self):
        """Fetches the list of qwen_image LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_qwen_image_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST QWEN IMAGE LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_qwen_image_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sd15_loras(self):
        """Fetches the list of sdxl LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_sd15_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST SD15 LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sd15l_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_controlnets(self):
        """Fetches the list of sdxl controlnets from the server."""
        url = f"http://{self.base_url}/list_sdxl_controlnets"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST SDXL CONTROLNETS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sdxl_controlnets ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_loras(self):
        """Fetches the list of sdxl LoRA filenames from the server."""
        url = f"http://{self.base_url}/list_sdxl_loras"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST SDXL LORAS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sdxl_loras ERROR: {e}")
            return {"ERROR": str(e)}

    async def list_sdxl_schedulers(self):
        """Fetches the list of sdxl schedulers from the server."""
        url = f"http://{self.base_url}/list_sdxl_schedulers"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LIST SDXL SCHEDULERS ERROR: {response.status_code}, Response: {response.text}")
                return {"ERROR": response.text}
        except Exception as e:
            print(f"list_sdxl_schedulers ERROR: {e}")
            return {"ERROR": str(e)}

    async def llm_chat(self, prompt, model_name=None, messages=None):
        """This takes a prompt, and optionally a model name and chat history, then returns a response"""
        url = f"http://{self.base_url}/llm_chat"
        data = {"prompt": prompt, "model_name": model_name, "messages": messages}

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LLM ERROR: {response.status_code}, Response: {response.text}")
                return {"LLM ERROR": response.text}
        except Exception as e:
            print(f"EXCEPTION ERROR: {e}")
            return {"ERROR": str(e)}

    async def ltx_ti2v(self, prompt, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, image=None,  seed=None, model_name=None, frame_rate=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/ltx_ti2v_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "image": image,
                "model_name": model_name,
                "frame_rate": frame_rate,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"LTX TI2V ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def lumina2_image(self, prompt, negative_prompt=None, model_name=None, width=None, height=None, steps=None,
                            batch_size=None, seed=None, guidance_scale=None, lora_name=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/lumina2_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"LUMINA2 ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_image(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None,
                               width=None, height=None, steps=None, batch_size=None, strength=None, seed=None,
                               true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_nunchaku_image(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None,
                                        width=None, height=None, steps=None, batch_size=None, strength=None, seed=None,
                                        true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_nunchaku_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE NUNCHAKU ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_inpaint_image(self, prompt, negative_prompt=None, image=None, model_name=None, width=None,
                                       height=None, steps=None, batch_size=None, true_cfg_scale=None, mask_image=None,
                                       strength=None, lora_name=None, seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_inpaint_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "true_cfg_scale": true_cfg_scale,
                "mask_image": mask_image,
                "strength": strength,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE INPAINT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_inpaint_nunchaku_image(self, prompt, negative_prompt=None, image=None, model_name=None, width=None,
                                                height=None, steps=None, batch_size=None, true_cfg_scale=None, mask_image=None,
                                                strength=None, lora_name=None, seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_inpaint_nunchaku_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "true_cfg_scale": true_cfg_scale,
                "mask_image": mask_image,
                "strength": strength,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE INPAINT NUNCHAKU ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_edit(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None,
                              width=None, height=None, steps=None, batch_size=None, seed=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_edit_generate"
        data = {"prompt": prompt,
                "negative_prompt":  negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE EDIT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_edit_nunchaku(self, prompt, negative_prompt=None, image=None, model_name=None, lora_name=None,
                                       width=None, height=None, steps=None, batch_size=None, seed=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_edit_nunchaku_generate"
        data = {"prompt": prompt,
                "negative_prompt":  negative_prompt,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE EDIT NUNCHAKU ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_edit_plus(self, prompt, negative_prompt=None, images=None, model_name=None, lora_name=None,
                              width=None, height=None, steps=None, batch_size=None, seed=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_edit_plus_generate"
        data = {"prompt": prompt,
                "negative_prompt":  negative_prompt,
                "images": images,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE EDIT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def qwen_image_edit_plus_nunchaku(self, prompt, negative_prompt=None, images=None, model_name=None, lora_name=None,
                                            width=None, height=None, steps=None, batch_size=None, seed=None, true_cfg_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/qwen_image_edit_plus_nunchaku_generate"
        data = {"prompt": prompt,
                "negative_prompt":  negative_prompt,
                "images": images,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "true_cfg_scale": true_cfg_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"QWEN IMAGE EDIT NUNCHAKU ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def realesrgan(self, image, scale=None):
        """This takes an image and an optional scale of either 2, 4, or 8 and returns an upscaled image"""
        url = f"http://{self.base_url}/realesrgan_generate"
        data = {"image": image,
                "scale": scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"REALESRGAN ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def sana_sprint_image(self, prompt, max_timesteps=None, intermediate_timesteps=None, image=None,
                                model_name=None, lora_name=None, width=None, height=None, steps=None, batch_size=None,
                                strength=None, seed=None, guidance_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sana_sprint_generate"
        data = {"prompt": prompt,
                "max_timesteps": max_timesteps,
                "intermediate_timesteps": intermediate_timesteps,
                "image": image,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "strength": strength,
                "seed": seed,
                "guidance_scale": guidance_scale}
        if intermediate_timesteps is not None:
            data["intermediate_timesteps"] = intermediate_timesteps
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SANA SPRINT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def sd15_image(self, prompt, image=None, negative_prompt=None, model_name=None, lora_name=None, width=None,
                         height=None, steps=None, batch_size=None, guidance_scale=None, strength=None, scheduler=None,
                         seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sd15_generate"
        data = {"prompt": prompt,
                "image": image,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "scheduler": scheduler,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SD15 ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def sd15_inpaint_image(self, prompt, image=None, negative_prompt=None, model_name=None, width=None,
                                 height=None, steps=None, batch_size=None, guidance_scale=None, mask_image=None,
                                 strength=None, lora_name=None, scheduler=None, seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sd15_inpaint_generate"
        data = {"prompt": prompt,
                "image": image,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "mask_image": mask_image,
                "strength": strength,
                "scheduler": scheduler,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SD15 INPAINT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def sdxl_image(self, prompt, image=None, negative_prompt=None, model_name=None, lora_name=None, width=None,
                         height=None, steps=None, batch_size=None, guidance_scale=None, strength=None,
                         controlnet_image=None, controlnet_processor=None, controlnet_conditioning=None,
                         ip_adapter_image=None, ip_adapter_strength=None, scheduler=None, seed=None):
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
                "guidance_scale": guidance_scale,
                "strength": strength,
                "controlnet_image": controlnet_image,
                "controlnet_processor": controlnet_processor,
                "controlnet_conditioning": controlnet_conditioning,
                "ip_adapter_strength": ip_adapter_strength,
                "ip_adapter_image": ip_adapter_image,
                "scheduler": scheduler,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SDXL ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def sdxl_inpaint_image(self, prompt, image=None, negative_prompt=None, model_name=None, width=None,
                                 height=None, steps=None, batch_size=None, guidance_scale=None, mask_image=None,
                                 strength=None, lora_name=None, scheduler=None, seed=None):
        """This takes a prompt, and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/sdxl_inpaint_generate"
        data = {"prompt": prompt,
                "image": image,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "guidance_scale": guidance_scale,
                "mask_image": mask_image,
                "strength": strength,
                "scheduler": scheduler,
                "seed": seed}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SDXL INPAINT ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def swin2sr(self, image):
        """This takes an image and returns an upscaled image"""
        url = f"http://{self.base_url}/swin2sr_generate"
        data = {"image": image}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"SWIN2SR ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def wan_ti2v(self, prompt, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, image=None,  seed=None, model_name=None, flow_shift=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/wan_ti2v_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "image": image,
                "model_name": model_name,
                "flow_shift": flow_shift,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"WAN TI2V ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def wan_vace(self, prompt, negative_prompt=None, width=None, height=None, steps=None, num_frames=None,
                       guidance_scale=None, first_frame=None, last_frame=None, flow_shift=None, seed=None,
                       model_name=None, lora_name=None):
        """This takes a prompt and returns a video"""
        url = f"http://{self.base_url}/wan_vace_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "steps": steps,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "flow_shift": flow_shift,
                "model_name": model_name,
                "lora_name": lora_name}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
                status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"WAN VACE ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def wan_v2v(self, prompt, negative_prompt=None, width=None, height=None, steps=None,
                      guidance_scale=None, seed=None, model_name=None, video_path=None, flow_shift=None, lora_name=None):
        """This takes a prompt and (optionally) a video, and returns a generated video."""
        url = f"http://{self.base_url}/wan_v2v_generate"
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "steps": steps,
            "model_name": model_name,
            "flow_shift": flow_shift,
            "lora_name": lora_name}
        data = {k: str(v) for k, v in data.items() if v is not None}
        files = None
        if video_path:
            async with aiofiles.open(video_path, "rb") as f:
                video_bytes = await f.read()
            files = {"video": (os.path.basename(video_path), video_bytes, "video/mp4")}

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, data=data, files=files)
            status_header = response.headers.get("x-status")
            if response.status_code == 200:
                return {"status": status_header,
                        "video": response.content}
            else:
                print(f"WAN TI2V ERROR: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}

    async def zimage_image(self, prompt, negative_prompt=None, model_name=None, lora_name=None, width=None,
                           height=None, steps=None, batch_size=None, seed=None, guidance_scale=None):
        """This takes a prompt and optional other variables and returns a list of base64 encoded images"""
        url = f"http://{self.base_url}/zimage_generate"
        data = {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "lora_name": lora_name,
                "width": width,
                "height": height,
                "steps": steps,
                "batch_size": batch_size,
                "seed": seed,
                "guidance_scale": guidance_scale}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ZIMAGE ERROR: {response.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")
            return {"ERROR": str(e)}


    async def update_url(self, url, port=6969):
        self.url = url
        self.port = port
        self.base_url = f"{self.url}:{self.port}"