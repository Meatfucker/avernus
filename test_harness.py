import time
from loguru import logger

from modules.tests import (ACETest,
                           CheckStatus,
                           ChromaTest, ChromaI2ITest,
                           FluxI2ITest, FluxInpaintTest, FluxIPAdapterTest, FluxKontextTest,
                           FluxLoraI2ITest, FluxLoraListTest, FluxLoraTest, FluxTest,
                           FramepackTest,
                           HiDreamTest,
                           HunyuanVideoTest,
                           ImageGenAuxTest,
                           Kandinsky5Test,
                           LlmChatTest,
                           QwenImageTest, QwenImageInpaintTest, QwenImageI2ITest, QwenImageEditTest,
                           RealESRGANTest,
                           SanaSprintTest, SanaSprintI2ITest,
                           SD15Test, SD15I2ITest, SD15InpaintTest,
                           SDXLControlnetI2ILoraTest, SDXLControlnetI2ITest, SDXLControlnetListTest,
                           SDXLControlnetLoraTest, SDXLControlnetTest, SDXLI2ILoraTest, SDXLI2ITest, SDXLInpaintTest,
                           SDXLIPAdapterTest, SDXLLoraListTest, SDXLLoraTest, SDXLTest, SDXLSchedulerListTest,
                           Swin2SRTest,
                           WANTest, WANI2VTest)

start_time = time.time()

#ace = ACETest("ACE STEP")

#check_status = CheckStatus("CHECK STATUS")

#chroma = ChromaTest("CHROMA")
#chroma_i2i = ChromaI2ITest("CHROMA_I2I")

#flux_lora_list = FluxLoraListTest("FLUX LORA LIST")
#flux = FluxTest("FLUX")
#flux_lora = FluxLoraTest("FLUX LORA")
#flux_i2i = FluxI2ITest("FLUX I2I")
#flux_inpaint = FluxInpaintTest("FLUX INPAINT")
#flux_lora_i2i = FluxLoraI2ITest("FLUX I2I LORA")
#flux_ipadapter = FluxIPAdapterTest("FLUX IP ADAPTER")
#flux_kontext_test = FluxKontextTest("FLUX KONTEXT")

#framepack = FramepackTest("FRAMEPACK")

#hidream = HiDreamTest("HIDREAM")

#hunyuan_t2v = HunyuanVideoTest("HUNYUAN_T2V")

#image_gen_aux = ImageGenAuxTest("IMAGE_GEN_AUX")

#kandinsky5 = Kandinsky5Test("KANDINSKY5_T2V")

#llm = LlmChatTest("LLM CHAT")

#qwen = QwenImageTest("QWEN_IMAGE")
#qwen_i2i = QwenImageI2ITest("QWEN IMAGE I2I")
#qwen_inpaint = QwenImageInpaintTest("QWEN IMAGE I2I")
#qwen_edit = QwenImageEditTest("QWEN IMAGE EDIT")

#realesrgan = RealESRGANTest("REALESRGAN")

sana_sprint = SanaSprintTest("SANA SPRINT")
sana_sprint_i2i = SanaSprintI2ITest("SANA SPRINT I2I")

#sd15 = SD15Test("SD15")
#sd15_i2i = SD15I2ITest("SD15_I2I")
#sd15_inpaint = SD15InpaintTest("SD15_INPAINT")

#sdxl_lora_list = SDXLLoraListTest("SDXL LORA LIST")
#sdxl_controlnet_list = SDXLControlnetListTest("SDXL CONTROLNET LIST")
#sdxl_scheduler_list = SDXLSchedulerListTest("SDXL SCHEDULERS LIST")
#sdxl = SDXLTest("SDXL")
#sdxl_lora = SDXLLoraTest("matlighty man. Watercolor backgrounds and cel-shaded characters, ethereal lighting,"
#                         " fantastical natural environments, whimsical yet haunting.",
#                         ["DarkGhibliMJ7SDXL.safetensors", "lighty.safetensors"], "SDXL LORA")
#sdxl_i2i = SDXLI2ITest("SDXL I2I")
#sdxl_inpaint = SDXLInpaintTest("SDXL INPAINT")
#sdxl_lora_i2i = SDXLI2ILoraTest("SDXL I2I LORA")
#sdxl_ipadapter = SDXLIPAdapterTest("SDXL IP ADAPTER")
#sdxl_control = SDXLControlnetTest("SDXL CONTROLNET")
#sdxl_control_lora = SDXLControlnetLoraTest("SDXL CONTROLNET LORA")
#sdxl_control_i2i = SDXLControlnetI2ITest("SDXL CONTROLNET I2I")
#sdxl_control_i2i_lora = SDXLControlnetI2ILoraTest("SDXL CONTROLNET I2I LORA")

#swin2sr = Swin2SRTest("SWIN2SR")

#wan = WANTest("WAN")
#wan_i2v = WANI2VTest("WAN_I2V")

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Total runtime: {elapsed_time:.2f} seconds")
