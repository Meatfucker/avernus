import asyncio
import base64
from io import BytesIO
import tempfile
import time

from loguru import logger
from PIL import Image

from modules.avernus_client import AvernusClient


class TimedTest:
    def __init__(self, test_name):
        self.client: AvernusClient = AvernusClient("127.0.0.1", "6969")
        self.test_name = test_name
        self._run_sync()

    def _run_sync(self):
        asyncio.run(self.run())

    async def run(self):
        logger.info(f'Testing Avernus {self.test_name}')
        start_time = time.time()
        try:
            status = await self.run_test()
            if status == "True" or status is True:
                logger.success(f"{self.test_name} SUCCESS:")
            else:
                logger.error(f"{self.test_name} FAIL")
        except Exception as e:
            logger.error(f"{self.test_name} FAIL: {e}")
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Total {self.test_name} runtime: {elapsed_time:.2f} seconds")

    async def run_test(self):
        pass

class ACETest(TimedTest):
    async def run_test(self):
        response = await self.client.ace_music(prompt="jazz, bass, techno", lyrics="inst", audio_duration=30)
        return response["status"]

class AuraflowTest(TimedTest):
    async def run_test(self):
        response = await self.client.auraflow_image("laser turtle", batch_size=1)
        await base64_image_to_file(response["images"], "auraflow")
        return response["status"]

class CheckStatus(TimedTest):
    async def run_test(self):
        status = await self.client.check_status()
        logger.success(status)
        return status["status"]

class ChromaTest(TimedTest):
    async def run_test(self):
        response = await self.client.chroma_image("frog made of yarn", batch_size=1)
        await base64_image_to_file(response["images"], "chroma")
        return response["status"]

class ChromaI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.chroma_image("frog made of yarn", image=image, batch_size=1)
        await base64_image_to_file(response["images"], "chroma_i2i")
        return response["status"]

class ChromaLoraListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_chroma_loras()
        logger.success(response["loras"])
        return response["status"]

class ChronoEditTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.chronoedit("make the balloon pop",
                                                image=image,
                                                batch_size=1)
        await base64_image_to_file(response["images"], "chronoedit")
        return response["status"]

class FluxI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.flux_image("basketball",
                                                image=image,
                                                batch_size=1,
                                                strength=0.7)
        await base64_image_to_file(response["images"], "flux_i2i")
        return response["status"]

class FluxKontextTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.flux_kontext("turn the bear into a gigiantic sandworm",
                                                  image=image,
                                                  batch_size=1)
        await base64_image_to_file(response["images"], "flux_kontext")
        return response["status"]

class FluxIPAdapterTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.flux_image("marijuana leaf",
                                                ip_adapter_image=image,
                                                ip_adapter_strength=0.6,
                                                batch_size=1)
        await base64_image_to_file(response["images"], "flux_ipadapter")
        return response["status"]

class FluxLoraI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.flux_image("matlighty man monstrous creature",
                                                image=image,
                                                batch_size=1,
                                                strength=0.9,
                                                lora_name="lighty_peft.safetensors")
        await base64_image_to_file(response["images"], "flux_lora_i2i")
        return response["status"]

class FluxInpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_ipadapter_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        response = await self.client.flux_inpaint_image("green beans",
                                                        batch_size=2,
                                                        image=image,
                                                        mask_image=mask_image,
                                                        strength=0.9)
        await base64_image_to_file(response["images"], "flux_inpaint")
        return response["status"]

class FluxLoraListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_flux_loras()
        logger.success(response["loras"])
        return response["status"]

class FluxLoraTest(TimedTest):
    async def run_test(self):
        response = await self.client.flux_image("man with a tattoo on his forehead",
                                              batch_size=1,
                                              lora_name="lighty_peft.safetensors")
        await base64_image_to_file(response["images"], "flux_lora")
        return response["status"]

class FluxTest(TimedTest):
    async def run_test(self):
        response = await self.client.flux_image("Mucus Balloon", batch_size=1)
        await base64_image_to_file(response["images"], "flux")
        return response["status"]

class FramepackTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        last_image = Image.open("tests/flux_inpaint_image_0.png")
        last_image = image_to_base64(last_image)
        response = await self.client.framepack("A colorful psychedelic scene of a mushroom with demons crawling on it",
                                               num_frames=33,
                                               seed=42,
                                               width=512,
                                               height=512,
                                               image=image,
                                               last_image=last_image)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]

class HiDreamTest(TimedTest):
    async def run_test(self):
        response = await self.client.hidream_image("laser turtle", batch_size=1)
        await base64_image_to_file(response["images"], "hidream")
        return response["status"]

class HunyuanVideoTest(TimedTest):
    async def run_test(self):
        response = await self.client.hunyuan_ti2v("An overhead tracking shot of frogs hopping from lilypad to lilypad, the water splashing as they land",
                                                  num_frames=33,
                                                  seed=42,
                                                  width=832,
                                                  height=480)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]

class ImageGenAuxTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.image_gen_aux_upscale(image=image, model="OzzyGT/DAT_X4")
        await base64_image_to_file(response["images"], "image_gen_aux")
        return response["status"]

class Kandinsky5Test(TimedTest):
    async def run_test(self):
        response = await self.client.kandinsky5_t2v("An overhead tracking shot of frogs hopping from lilypad to lilypad, the water splashing as they land",
                                                    num_frames=33,
                                                    seed=42,
                                                    width=768,
                                                    height=512)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]


class LlmChatTest(TimedTest):
    async def run_test(self):
        messages = [{"role": "user", "content": "The best color of dick is purple"},
                    {"role": "assistant", "content": "The best color of dick is purple"}]
        response = await self.client.llm_chat("What was the best color again and what was the thing that was that color?",
                                         "Meatfucker/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4-bnb-nf4",
                                         messages)
        logger.success(response["response"])
        return response["status"]

class LTXTest(TimedTest):
    async def run_test(self):
        response = await self.client.ltx_ti2v("An overhead tracking shot of frogs hopping from lilypad to lilypad, the water splashing as they land",
                                   num_frames=33,
                                   seed=42,
                                   width=704,
                                   height=512)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]

class Lumina2Test(TimedTest):
    async def run_test(self):
        response = await self.client.lumina2_image("laser turtle", batch_size=1)
        await base64_image_to_file(response["images"], "lumina2")
        return response["status"]

class Lumina2SingleFileTest(TimedTest):
    async def run_test(self):
        response = await self.client.lumina2_image("laser turtle",
                                                   batch_size=1,
                                                   model_name="https://huggingface.co/neta-art/Neta-Lumina/blob/main/neta-lumina-v1.0-all-in-one.safetensors")
        await base64_image_to_file(response["images"], "lumina2_single")
        return response["status"]

class QwenImageTest(TimedTest):
    async def run_test(self):
        response = await self.client.qwen_image_image("redheaded woman wearing green dress", batch_size=1)
        await base64_image_to_file(response["images"], "qwen_image")
        return response["status"]

class QwenImageEditTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.qwen_image_edit("turn the bear into a gigiantic sandworm",
                                                     image=image,
                                                     batch_size=1)
        await base64_image_to_file(response["images"], "qwen_image_edit")
        return response["status"]


class QwenImageI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        response = await self.client.qwen_image_image("basketball",
                                                      image=image,
                                                      batch_size=1,
                                                      strength=0.7)
        await base64_image_to_file(response["images"], "qwen_image_i2i")
        return response["status"]

class QwenImageInpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_ipadapter_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        response = await self.client.flux_inpaint_image("green beans",
                                                        batch_size=2,
                                                        image=image,
                                                        mask_image=mask_image,
                                                        strength=0.9)
        await base64_image_to_file(response["images"], "qwen_image_inpaint")
        return response["status"]

class QwenImageLoraListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_qwen_image_loras()
        logger.success(response["loras"])
        return response["status"]


class RealESRGANTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.realesrgan(image=image,
                                                scale=8)
        await base64_image_to_file(response["images"], "realesrgan")
        return response["status"]

class SanaSprintTest(TimedTest):
    async def run_test(self):
        response = await self.client.sana_sprint_image("frog made of yarn", batch_size=1)
        await base64_image_to_file(response["images"], "sana_sprint")
        return response["status"]

class SanaSprintI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.sana_sprint_image("frog made of yarn", image=image, batch_size=1)
        await base64_image_to_file(response["images"], "sana_sprint_i2i")
        return response["status"]

class SD15Test(TimedTest):
    async def run_test(self):
        response = await self.client.sd15_image("turbo taco",
                                                batch_size=2,
                                                scheduler="DPMSolverSDEScheduler")
        await base64_image_to_file(response["images"], "sd15")
        return response["status"]

class SD15I2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        response = await self.client.sd15_image("warrior princess",
                                                image=image,
                                                strength=0.3,
                                                batch_size=2)
        await base64_image_to_file(response["images"], "sd15_i2i")
        return response["status"]

class SD15InpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        response = await self.client.sd15_inpaint_image("cartoon",
                                                        batch_size=2,
                                                        image=image,
                                                        mask_image=mask_image,
                                                        strength=1)
        await base64_image_to_file(response["images"], "sd15_inpaint")
        return response["status"]

class SD15LoraListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_sd15_loras()
        logger.success(response["loras"])
        return response["status"]


class SDXLControlnetI2ILoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/sdxl_image_0.png")
        controlnet_image = image_to_base64(controlnet_image)
        response = await self.client.sdxl_image("jello monster",
                                                batch_size=2,
                                                image=image,
                                                lora_name="lighty.safetensors",
                                                controlnet_image=controlnet_image,
                                                controlnet_processor="depth",
                                                controlnet_conditioning=0.5)
        await base64_image_to_file(response["images"], "sdxl_controlnet_i2i_lora")
        return response["status"]

class SDXLControlnetI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/sdxl_image_0.png")
        controlnet_image = image_to_base64(controlnet_image)
        response = await self.client.sdxl_image("cartoon",
                                                batch_size=2,
                                                image=image,
                                                controlnet_image=controlnet_image,
                                                controlnet_processor="canny",
                                                controlnet_conditioning=0.5)
        await base64_image_to_file(response["images"], "sdxl_controlnet_i2i")
        return response["status"]

class SDXLControlnetListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_sdxl_controlnets()
        logger.success(response["sdxl_controlnets"])
        return response["status"]

class SDXLControlnetLoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        response = await self.client.sdxl_image("wood statue",
                                                batch_size=2,
                                                lora_name="lighty.safetensors",
                                                controlnet_image=image,
                                                controlnet_processor="canny",
                                                controlnet_conditioning=0.5)
        await base64_image_to_file(response["images"], "sdxl_controlnet_lora")
        return response["status"]

class SDXLControlnetTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        response = await self.client.sdxl_image("statue",
                                                batch_size=2,
                                                controlnet_image=image,
                                                controlnet_processor="canny",
                                                controlnet_conditioning=0.5)
        await base64_image_to_file(response["images"], "sdxl_controlnet")
        return response["status"]

class SDXLI2ILoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        response = await self.client.sdxl_image("matlighty man robot hooker",
                                                image=image,
                                                strength=0.7,
                                                lora_name="lighty.safetensors",
                                                batch_size=2)
        await base64_image_to_file(response["images"], "sdxl_i2i_lora")
        return response["status"]

class SDXLI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        response = await self.client.sdxl_image("warrior princess",
                                                image=image,
                                                strength=0.3,
                                                batch_size=2)
        await base64_image_to_file(response["images"], "sdxl_i2i")
        return response["status"]

class SDXLInpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        response = await self.client.sdxl_inpaint_image("cartoon",
                                                        batch_size=2,
                                                        image=image,
                                                        mask_image=mask_image,
                                                        strength=1)
        await base64_image_to_file(response["images"], "sdxl_inpaint")
        return response["status"]

class SDXLIPAdapterTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.sdxl_image("marijuana leaf",
                                                ip_adapter_image=image,
                                                ip_adapter_strength=0.6,
                                                batch_size=2)
        await base64_image_to_file(response["images"], "sdxl_ipadapter")
        return response["status"]

class SDXLLoraListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_sdxl_loras()
        logger.success(response["loras"])
        return response["status"]

class SDXLLoraTest(TimedTest):
    def __init__(self, prompt, lora_name, test_name):
        self.prompt = prompt
        self.lora_name = lora_name
        super().__init__(test_name)

    async def run_test(self):
        if isinstance(self.lora_name, str):
            self.lora_name = [self.lora_name]

        response = await self.client.sdxl_image(self.prompt,
                                                lora_name=self.lora_name,
                                                batch_size=2)
        await base64_image_to_file(response["images"], "sdxl_lora")
        return response["status"]

class SDXLSchedulerListTest(TimedTest):
    async def run_test(self):
        response = await self.client.list_sdxl_schedulers()
        logger.success(response["schedulers"])
        return response["status"]

class SDXLTest(TimedTest):
    async def run_test(self):
        response = await self.client.sdxl_image("wizard",
                                                batch_size=2,
                                                scheduler="DPMSolverSDEScheduler")
        await base64_image_to_file(response["images"], "sdxl")
        return response["status"]

class Swin2SRTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.swin2sr(image=image)
        await base64_image_to_file(response["images"], "swin2sr")
        return response["status"]

class WANTest(TimedTest):
    async def run_test(self):
        response = await self.client.wan_ti2v("An overhead tracking shot of frogs hopping from lilypad to lilypad, the water splashing as they land",
                                              num_frames=33,
                                              seed=42,
                                              width=832,
                                              height=480)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]

class WANI2VTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        response = await self.client.wan_ti2v("A colorful psychedelic scene of a mushroom with demons crawling on it",
                                              num_frames=81,
                                              seed=42,
                                              width=832,
                                              height=480,
                                              image=image)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response["video"])
        temp_file.close()
        print(temp_file.name)
        return response["status"]

async def base64_image_to_file(base64_images, prefix=""):
    for i, b64_img in enumerate(base64_images):
        img_data = base64.b64decode(b64_img)  # Decode base64 string
        filename = f"tests/{prefix}_image_{i}.png"  # Create a unique filename
        with open(filename, "wb") as f:
            f.write(img_data)  # Write binary data to file

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
