import time
from loguru import logger

from modules.tests import (ACETest, CheckStatus, FluxControlnetI2ILoraTest, FluxControlnetI2ITest,
                           FluxControlnetListTest, FluxControlnetLoraTest, FluxControlnetTest, FluxI2ITest,
                           FluxInpaintTest, FluxIPAdapterTest, FluxKontextTest, FluxLoraI2ITest, FluxLoraListTest,
                           FluxLoraTest, FluxTest, LlmChatTest, LlmRagTest, LTXTest, QwenImageTest,
                           SDXLControlnetI2ILoraTest, SDXLControlnetI2ITest, SDXLControlnetListTest,
                           SDXLControlnetLoraTest, SDXLControlnetTest, SDXLI2ILoraTest, SDXLI2ITest, SDXLInpaintTest,
                           SDXLIPAdapterTest, SDXLLoraListTest, SDXLLoraTest, SDXLTest, SDXLSchedulerListTest, WANTest)
start_time = time.time()

check_status = CheckStatus("CHECK STATUS")
#llm = LlmChatTest("LLM CHAT")
#rag = LlmRagTest("LLM RAG")
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

#flux_lora_list = FluxLoraListTest("FLUX LORA LIST")
#flux_control_list = FluxControlnetListTest("FLUX CONTROLNET LIST")
#flux = FluxTest("FLUX")
#flux_lora = FluxLoraTest("FLUX LORA")
#flux_i2i = FluxI2ITest("FLUX I2I")
#flux_inpaint = FluxInpaintTest("FLUX INPAINT")
#flux_lora_i2i = FluxLoraI2ITest("FLUX I2I LORA")
#flux_ipadapter = FluxIPAdapterTest("FLUX IP ADAPTER")
#flux_control = FluxControlnetTest("FLUX CONTROLNET")
#flux_control_lora = FluxControlnetLoraTest("FLUX CONTROLNET LORA")
#flux_control_i2i = FluxControlnetI2ITest("FLUX CONTROLNET I2I")
#flux_control_i2i_lora = FluxControlnetI2ILoraTest("FLUX CONTROLNET I2I LORA")
#flux_kontext_test = FluxKontextTest("FLUX KONTEXT")

#ltx = LTXTest("LTX")
#wan = WANTest("WAN")
qwen = QwenImageTest("QWEN_IMAGE")
#ace = ACETest("ACE STEP")
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Total runtime: {elapsed_time:.2f} seconds")
