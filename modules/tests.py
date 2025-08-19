import asyncio
import base64
from io import BytesIO
import time
from loguru import logger
from modules.avernus_client import AvernusClient
from PIL import Image

class TimedTest:
    def __init__(self, test_name):
        self.client = client = AvernusClient("127.0.0.1", "6969")
        self.test_name = test_name
        self._run_sync()

    def _run_sync(self):
        asyncio.run(self.run())

    async def run(self):
        logger.info(f'Testing Avernus {self.test_name}')
        start_time = time.time()
        try:
            await self.run_test()
            logger.success(f"{self.test_name} SUCCESS:")
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
        await self.client.ace_music(prompt="jazz, bass, techno", lyrics="inst")

class CheckStatus(TimedTest):
    async def run_test(self):
        status = await self.client.check_status()
        logger.success(status)

class FluxControlnetI2ILoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/mushroom.png")
        controlnet_image = image_to_base64(controlnet_image)
        images = await self.client.flux_image("jello monster man",
                                              batch_size=2,
                                              image=image,
                                              lora_name="lighty_peft.safetensors",
                                              controlnet_image=controlnet_image,
                                              controlnet_processor="canny")
        await base64_image_to_file(images, "flux_controlnet_i2i_lora")

class FluxControlnetI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/mushroom.png")
        controlnet_image = image_to_base64(controlnet_image)
        images = await self.client.flux_image("cartoon",
                                              batch_size=2,
                                              image=image,
                                              controlnet_image=controlnet_image,
                                              controlnet_processor="canny")
        await base64_image_to_file(images, "flux_controlnet_i2i")

class FluxControlnetListTest(TimedTest):
    async def run_test(self):
        controlnets = await self.client.list_flux_controlnets()
        logger.success(controlnets)

class FluxControlnetLoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.flux_image("wood statue",
                                              batch_size=2,
                                              lora_name="lighty_peft.safetensors",
                                              controlnet_image=image,
                                              controlnet_processor="canny")
        await base64_image_to_file(images, "flux_controlnet_lora")

class FluxControlnetTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.flux_image("statue",
                                              batch_size=2,
                                              controlnet_image=image,
                                              controlnet_processor="canny")
        await base64_image_to_file(images, "flux_controlnet")

class FluxI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        images = await self.client.flux_image("basketball",
                                              image=image,
                                              batch_size=1,
                                              strength=0.7)
        await base64_image_to_file(images, "flux_i2i")

class FluxKontextTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        images = await self.client.flux_kontext("turn the bear into a gigiantic sandworm",
                                                image=image,
                                                batch_size=1)
        await base64_image_to_file(images, "flux_kontext")

class FluxIPAdapterTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        images = await self.client.flux_image("marijuana leaf",
                                              ip_adapter_image=image,
                                              ip_adapter_strength=0.6,
                                              batch_size=2)
        await base64_image_to_file(images, "flux_ipadapter")

class FluxLoraI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        images = await self.client.flux_image("matlighty man monstrous creature",
                                              image=image,
                                              batch_size=1,
                                              strength=0.9,
                                              lora_name="lighty_peft.safetensors")
        await base64_image_to_file(images, "flux_lora_i2i")

class FluxInpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/flux_ipadapter_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        images = await self.client.flux_inpaint_image("green beans",
                                                      batch_size=2,
                                                      image=image,
                                                      mask_image=mask_image,
                                                      strength=0.9)
        await base64_image_to_file(images, "flux_inpaint")

class FluxLoraListTest(TimedTest):
    async def run_test(self):
        loras = await self.client.list_flux_loras()
        logger.success(loras)

class FluxLoraTest(TimedTest):
    async def run_test(self):
        images = await self.client.flux_image("man with a tattoo on his forehead",
                                              batch_size=1,
                                              lora_name="lighty_peft.safetensors")
        await base64_image_to_file(images, "flux_lora")

class FluxTest(TimedTest):
    async def run_test(self):
        images = await self.client.flux_image("Mucus Balloon", batch_size=1)
        await base64_image_to_file(images, "flux")

class LlmChatTest(TimedTest):
    async def run_test(self):
        messages = [{"role": "user", "content": "The best color of dick is purple"},
                    {"role": "assistant", "content": "The best color of dick is purple"}]
        response = await self.client.llm_chat("What was the best color again and what was the thing that was that color?",
                                         "Goekdeniz-Guelmez/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2",
                                         messages)
        logger.success(response)

class LlmRagTest(TimedTest):
    async def run_test(self):
        rag_results = await self.client.rag_retrieve("What is the Atlas of Worlds")
        for result in rag_results:
            logger.success(result)

class LTXTest(TimedTest):
    async def run_test(self):
        await self.client.ltx_video("frogs hopping")

class SDXLControlnetI2ILoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/sdxl_image_0.png")
        controlnet_image = image_to_base64(controlnet_image)
        images = await self.client.sdxl_image("jello monster",
                                              batch_size=2,
                                              image=image,
                                              lora_name="lighty.safetensors",
                                              controlnet_image=controlnet_image,
                                              controlnet_processor="depth",
                                              controlnet_conditioning=0.5)
        await base64_image_to_file(images, "sdxl_controlnet_i2i_lora")

class SDXLControlnetI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        controlnet_image = Image.open("tests/sdxl_image_0.png")
        controlnet_image = image_to_base64(controlnet_image)
        images = await self.client.sdxl_image("cartoon",
                                              batch_size=2,
                                              image=image,
                                              controlnet_image=controlnet_image,
                                              controlnet_processor="canny",
                                              controlnet_conditioning=0.5)
        await base64_image_to_file(images, "sdxl_controlnet_i2i")

class SDXLControlnetListTest(TimedTest):
    async def run_test(self):
        controlnets = await self.client.list_sdxl_controlnets()
        logger.success(controlnets)

class SDXLControlnetLoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.sdxl_image("wood statue",
                                              batch_size=2,
                                              lora_name="lighty.safetensors",
                                              controlnet_image=image,
                                              controlnet_processor="canny",
                                              controlnet_conditioning=0.5)
        await base64_image_to_file(images, "sdxl_controlnet_lora")

class SDXLControlnetTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.sdxl_image("statue",
                                              batch_size=2,
                                              controlnet_image=image,
                                              controlnet_processor="canny",
                                              controlnet_conditioning=0.5)
        await base64_image_to_file(images, "sdxl_controlnet")

class SDXLI2ILoraTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.sdxl_image("matlighty man robot hooker",
                                              image=image,
                                              strength=0.7,
                                              lora_name="lighty.safetensors",
                                              batch_size=2)
        await base64_image_to_file(images, "sdxl_i2i_lora")

class SDXLI2ITest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await self.client.sdxl_image("warrior princess",
                                              image=image,
                                              strength=0.3,
                                              batch_size=2)
        await base64_image_to_file(images, "sdxl_i2i")

class SDXLInpaintTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/sdxl_i2i_image_0.png")
        image = image_to_base64(image)
        mask_image = Image.open("tests/mask_image.png")
        mask_image = image_to_base64(mask_image)
        images = await self.client.sdxl_inpaint_image("cartoon",
                                                      batch_size=2,
                                                      image=image,
                                                      mask_image=mask_image,
                                                      strength=1)
        await base64_image_to_file(images, "sdxl_inpaint")

class SDXLIPAdapterTest(TimedTest):
    async def run_test(self):
        image = Image.open("tests/mushroom.png")
        image = image_to_base64(image)
        images = await self.client.sdxl_image("marijuana leaf",
                                              ip_adapter_image=image,
                                              ip_adapter_strength=0.6,
                                              batch_size=2)
        await base64_image_to_file(images, "sdxl_ipadapter")

class SDXLLoraListTest(TimedTest):
    async def run_test(self):
        loras = await self.client.list_sdxl_loras()
        logger.success(loras)

class SDXLLoraTest(TimedTest):
    def __init__(self, prompt, lora_name, test_name):
        self.prompt = prompt
        self.lora_name = lora_name
        super().__init__(test_name)

    async def run_test(self):
        if isinstance(self.lora_name, str):
            self.lora_name = [self.lora_name]

        images = await self.client.sdxl_image(self.prompt,
                                              lora_name=self.lora_name,
                                              batch_size=2)
        await base64_image_to_file(images, "sdxl_lora")

class SDXLSchedulerListTest(TimedTest):
    async def run_test(self):
        schedulers = await self.client.list_sdxl_schedulers()
        logger.success(schedulers)

class SDXLTest(TimedTest):
    async def run_test(self):
        images = await self.client.sdxl_image("wizard",
                                              batch_size=2,
                                              scheduler="DPMSolverSDEScheduler")
        await base64_image_to_file(images, "sdxl")

class WANTest(TimedTest):
    async def run_test(self):
        await self.client.wan_video("frogs hopping", num_frames=33)

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