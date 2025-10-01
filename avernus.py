from fastapi import FastAPI, Body
from loguru import logger

from modules.pydantic_models import (ACEStepRequest, FluxInpaintRequest, FluxRequest, FluxResponse,
                                     FluxLoraListResponse, LLMRequest, LLMResponse, QwenImageRequest, QwenImageInpaintRequest,
                                     QwenImageLoraListResponse, QwenImageResponse, SDXLInpaintRequest, SDXLRequest,
                                     SDXLResponse, SDXLLoraListResponse, SDXLControlnetListResponse, StatusResponse,
                                     SDXLSchedulerListResponse, WanTI2VRequest)
from modules.utils import ServerManager, return_loras, forward_post_request, forward_stream_request, setup_logging

setup_logging()
avernus = FastAPI()
PIPELINE = None
server_manager = ServerManager()


@avernus.post("/ace_generate")
async def ace_generate(data: ACEStepRequest = Body(...)):
    """Generates audio based on user inputs"""
    logger.info("ace_generate request received")
    await server_manager.set_pipeline("ace", "ACE")
    url = "http://127.0.0.1:6970/ace_generate"
    return await forward_stream_request(url, data)

@avernus.post("/flux_generate", response_model=FluxResponse)
async def flux_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
    logger.info("flux_generate request received")
    await server_manager.set_pipeline("flux", "FLUX.1-Krea-dev")
    url = "http://127.0.0.1:6970/flux_generate"
    return await forward_post_request(url, data)

@avernus.post("/flux_inpaint_generate", response_model=FluxResponse)
async def flux_inpaint_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux inpaint images based on user inputs."""
    logger.info("flux_inpaint_generate request received")
    await server_manager.set_pipeline("flux_inpaint", "FLUX.1-dev")
    url = "http://127.0.0.1:6970/flux_inpaint_generate"
    return await forward_post_request(url, data)

@avernus.post("/flux_fill_generate", response_model=FluxResponse)
async def flux_fill_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux fill images based on user inputs."""
    logger.info("flux_fill_generate request received")
    await server_manager.set_pipeline("flux_fill", "FLUX.1-Fill-dev")
    url = "http://127.0.0.1:6970/flux_fill_generate"
    return await forward_post_request(url, data)

@avernus.post("/flux_kontext_generate", response_model=FluxResponse)
async def flux_kontext_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux Kontext images based on user inputs"""
    logger.info("flux_kontext_generate request received")
    await server_manager.set_pipeline("flux_kontext", "FLUX.1-Kontext-dev")
    url = "http://127.0.0.1:6970/flux_kontext_generate"
    return await forward_post_request(url, data)

@avernus.get("/list_flux_loras", response_model=FluxLoraListResponse)
async def list_flux_loras():
    """Returns a list of the files located in the flux loras directory"""
    logger.info("list_flux_loras request received")
    try:
        filenames = return_loras("loras/flux")
        return {"loras": filenames}
    except Exception as e:
        logger.error(f"list_flux_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_qwen_image_loras", response_model=QwenImageLoraListResponse)
async def list_qwen_image_loras():
    """Returns a list of the files located in the qwen_image loras directory"""
    logger.info("list_qwen_image_loras request received")
    try:
        filenames = return_loras("loras/qwen_image")
        return {"loras": filenames}
    except Exception as e:
        logger.error(f"list_qwen_image_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_sdxl_controlnets", response_model=SDXLControlnetListResponse)
async def list_sdxl_controlnets():
    """Returns a list of available sdxl controlnets"""
    logger.info("list_sdxl_controlnets request received")
    try:
        return {"sdxl_controlnets": ["depth", "canny"]}
    except Exception as e:
        logger.error(f"list_sdxl_controlnets ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_sdxl_loras", response_model=SDXLLoraListResponse)
async def list_sdxl_loras():
    """Returns a list of the files located in the sdxl loras directory."""
    logger.info("list_sdxl_loras request received")
    try:
        filenames = return_loras("loras/sdxl")
        return {"loras": filenames}
    except Exception as e:
        logger.error(f"list_sdxl_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_sdxl_schedulers", response_model=SDXLSchedulerListResponse)
async def list_sdxl_schedulers():
    """Returns a list of sdxl schedulers."""
    logger.info("list_sdxl_schedulers request received")
    try:
        schedulers = {"schedulers": ["DPMSolverSinglestepScheduler",
                                     "DDIMScheduler",
                                     "DDPMScheduler",
                                     "PNDMScheduler",
                                     "LMSDiscreteScheduler",
                                     "EulerDiscreteScheduler",
                                     "HeunDiscreteScheduler",
                                     "EulerAncestralDiscreteScheduler",
                                     "DPMSolverMultistepScheduler",
                                     "KDPM2DiscreteScheduler",
                                     "KDPM2AncestralDiscreteScheduler",
                                     "DEISMultistepScheduler",
                                     "UniPCMultistepScheduler",
                                     "DPMSolverSDEScheduler"]}
        return schedulers
    except Exception as e:
        logger.error(f"list_sdxl_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.post("/llm_chat", response_model=LLMResponse)
async def llm_chat(data: LLMRequest = Body(...)):
    """This takes a prompt, and optionally a Huggingface model name, and/or a Huggingface formatted message history.
    See test_harness.py for an example of one"""
    logger.info("llm_chat request received")
    await server_manager.set_pipeline("chat", data.model_name)
    url = "http://127.0.0.1:6970/llm_chat"
    return await forward_post_request(url, data)

@avernus.post("/qwen_image_generate", response_model=QwenImageResponse)
async def qwen_image_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image images based on user inputs"""
    if data.image:
        logger.info("qwen_image_i2i_generate request received")
        await server_manager.set_pipeline("qwen_image_i2i", "Qwen-Image")
        url = "http://127.0.0.1:6970/qwen_image_i2i_generate"
    else:
        logger.info("qwen_image_generate request received")
        await server_manager.set_pipeline("qwen_image", "Qwen-Image")
        url = "http://127.0.0.1:6970/qwen_image_generate"
    return await forward_post_request(url, data)

@avernus.post("/qwen_image_inpaint_generate", response_model=QwenImageResponse)
async def qwen_image_inpaint_generate(data: QwenImageInpaintRequest = Body(...)):
    """Generates some number of qwen image inpaint images based on user inputs."""
    logger.info("qwen_image_inpaint_generate request received")
    await server_manager.set_pipeline("qwen_image_inpaint", "Qwen-Image")
    url = "http://127.0.0.1:6970/qwen_image_inpaint_generate"
    return await forward_post_request(url, data)

@avernus.post("/qwen_image_edit_generate", response_model=QwenImageResponse)
async def qwen_image_edit_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_generate request received")
    await server_manager.set_pipeline("qwen_image_edit", "Qwen-Image-Edit")
    url = "http://127.0.0.1:6970/qwen_image_edit_generate"
    return await forward_post_request(url, data)

@avernus.get("/status", response_model=StatusResponse)
async def status():
    """ This returns Ok when hit"""
    logger.info("status request received")
    return {"status": str("Ok!"), "version": str("0.6.0")}

@avernus.post("/sdxl_generate", response_model=SDXLResponse)
async def sdxl_generate(data: SDXLRequest = Body(...)):
    """Generates some number of sdxl images based on user inputs."""
    logger.info("sdxl_generate request received")
    if data.image is None and data.controlnet_image is None:
        await server_manager.set_pipeline("sdxl", data.model_name)
    if data.image is None and data.controlnet_image is not None:
        await server_manager.set_pipeline("sdxl_controlnet", data.model_name)
    if data.image is not None and data.controlnet_image is None:
        await server_manager.set_pipeline("sdxl_i2i", data.model_name)
    if data.image is not None and data.controlnet_image is not None:
        await server_manager.set_pipeline("sdxl_i2i_controlnet", data.model_name)
    url = "http://127.0.0.1:6970/sdxl_generate"
    return await forward_post_request(url, data)

@avernus.post("/sdxl_inpaint_generate", response_model=SDXLResponse)
async def sdxl_inpaint_generate(data: SDXLInpaintRequest = Body(...)):
    """Generates some number of sdxl inpaint images based on user inputs."""
    logger.info("sdxl_inpaint_generate request received")
    await server_manager.set_pipeline("sdxl_inpaint", data.model_name)
    url = "http://127.0.0.1:6970/sdxl_inpaint_generate"
    return await forward_post_request(url, data)

@avernus.post("/wan_ti2v_generate")
async def wan_ti2v_generate(data: WanTI2VRequest = Body(...)):
    logger.info("wan_ti2v_generate request received")
    if data.image is None:
        await server_manager.set_pipeline("wan", data.model_name)
    if data.image is not None:
        await server_manager.set_pipeline("wan_i2v", data.model_name)
    url = "http://127.0.0.1:6970/wan_ti2v_generate"
    return await forward_stream_request(url, data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus, host="0.0.0.0")