import asyncio
import os

from fastapi import FastAPI, Body, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from loguru import logger

from modules.pydantic_models import (ACEStepRequest,
                                     ChromaRequest, ChromaResponse, ChromaLoraListResponse,
                                     FluxInpaintRequest, FluxRequest, FluxResponse, FluxLoraListResponse,
                                     FramepackRequest,
                                     HiDreamResponse, HiDreamRequest,
                                     HunyuanTI2VRequest,
                                     ImageGenAuxRequest, ImageGenAuxResponse,
                                     LLMRequest, LLMResponse,
                                     QwenImageRequest, QwenImageInpaintRequest, QwenImageLoraListResponse,
                                     QwenImageResponse, QwenImageEditPlusRequest,
                                     RealESRGANResponse, RealESRGANRequest,
                                     SDXLInpaintRequest, SDXLRequest, SDXLResponse,
                                     SDXLLoraListResponse,
                                     SDXLControlnetListResponse, SDXLSchedulerListResponse,
                                     StatusResponse,
                                     Swin2SRResponse, Swin2SRRequest,
                                     WanTI2VRequest, WanVACERequest)
from modules.utils import (ServerManager, return_loras, forward_post_request, setup_logging,
                           cleanup_and_stream)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
setup_logging()
avernus = FastAPI()
PIPELINE = None
server_manager = ServerManager()
pipeline_lock = asyncio.Lock()


@avernus.post("/ace_generate")
async def ace_generate(data: ACEStepRequest = Body(...)):
    """Generates audio based on user inputs"""
    logger.info("ace_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("ace", "ACE")
        url = "http://127.0.0.1:6970/ace_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return StreamingResponse(open(result["path"], "rb"), media_type="audio/wav")
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/chroma_generate", response_model=ChromaResponse)
async def chroma_generate(data: ChromaRequest = Body(...)):
    """Generates some number of Chroma images based on user inputs"""
    logger.info("chroma_generate request received")
    async with pipeline_lock:
        if data.image is not None:
            await server_manager.set_pipeline("chroma_i2i", data.model_name)
        else:
            await server_manager.set_pipeline("chroma", data.model_name)
        url = "http://127.0.0.1:6970/chroma_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/flux_generate", response_model=FluxResponse)
async def flux_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
    logger.info("flux_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("flux", data.model_name)
        url = "http://127.0.0.1:6970/flux_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/flux_inpaint_generate", response_model=FluxResponse)
async def flux_inpaint_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux inpaint images based on user inputs."""
    logger.info("flux_inpaint_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("flux_inpaint", data.model_name)
        url = "http://127.0.0.1:6970/flux_inpaint_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/flux_fill_generate", response_model=FluxResponse)
async def flux_fill_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux fill images based on user inputs."""
    logger.info("flux_fill_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("flux_fill", "FLUX.1-Fill-dev")
        url = "http://127.0.0.1:6970/flux_fill_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/flux_kontext_generate", response_model=FluxResponse)
async def flux_kontext_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux Kontext images based on user inputs"""
    logger.info("flux_kontext_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("flux_kontext", "FLUX.1-Kontext-dev")
        url = "http://127.0.0.1:6970/flux_kontext_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/framepack_generate")
async def framepack_generate(data: FramepackRequest = Body(...)):
    logger.info("framepack_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("framepack", data.model_name)
        url = "http://127.0.0.1:6970/framepack_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return StreamingResponse(cleanup_and_stream(result["path"]), media_type="video/mp4")
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/hidream_generate", response_model=HiDreamResponse)
async def hidream_generate(data: HiDreamRequest = Body(...)):
    """Generates some number of HiDream images based on user inputs"""
    logger.info("hidream_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("hidream", data.model_name)
        url = "http://127.0.0.1:6970/hidream_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/hunyuan_ti2v_generate")
async def hunyuan_ti2v_generate(data: HunyuanTI2VRequest = Body(...)):
    logger.info("hunyuan_ti2v_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("hunyuan_t2v", data.model_name)
        url = "http://127.0.0.1:6970/hunyuan_ti2v_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return StreamingResponse(cleanup_and_stream(result["path"]), media_type="video/mp4")
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}


@avernus.post("/image_gen_aux_upscale", response_model=ImageGenAuxResponse)
async def image_gen_aux_upscale(data: ImageGenAuxRequest = Body(...)):
    """Upscales an image based on user input"""
    logger.info("image_gen_aux_upscale request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("image_gen_aux_upscale", data.model)
        url = "http://127.0.0.1:6970/image_gen_aux_upscale"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.get("/list_chroma_loras", response_model=ChromaLoraListResponse)
async def list_chroma_loras():
    """Returns a list of the files located in the flux loras directory"""
    logger.info("list_chroma_loras request received")
    try:
        filenames = return_loras("loras/chroma")
        return {"loras": filenames}
    except Exception as e:
        logger.error(f"list_chroma_loras ERROR: {e}")
        return {"error": str(e)}

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
    async with pipeline_lock:
        await server_manager.set_pipeline("chat", data.model_name)
        url = "http://127.0.0.1:6970/llm_chat"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_generate", response_model=QwenImageResponse)
async def qwen_image_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image images based on user inputs"""
    async with pipeline_lock:
        if data.image:
            logger.info("qwen_image_i2i_generate request received")
            await server_manager.set_pipeline("qwen_image_i2i", "Qwen-Image")
            url = "http://127.0.0.1:6970/qwen_image_i2i_generate"
        else:
            logger.info("qwen_image_generate request received")
            await server_manager.set_pipeline("qwen_image", "Qwen-Image")
            url = "http://127.0.0.1:6970/qwen_image_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_nunchaku_generate", response_model=QwenImageResponse)
async def qwen_image_nunchaku_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image Nunchaku images based on user inputs"""
    async with pipeline_lock:
        if data.image:
            logger.info("qwen_image_nunchaku_i2i_generate request received")
            await server_manager.set_pipeline("qwen_image_i2i_nunchaku", "Qwen-Image")
            url = "http://127.0.0.1:6970/qwen_image_i2i_nunchaku_generate"
        else:
            logger.info("qwen_image_nunchaku_generate request received")
            await server_manager.set_pipeline("qwen_image_nunchaku", "Qwen-Image")
            url = "http://127.0.0.1:6970/qwen_image_nunchaku_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_inpaint_generate", response_model=QwenImageResponse)
async def qwen_image_inpaint_generate(data: QwenImageInpaintRequest = Body(...)):
    """Generates some number of qwen image inpaint images based on user inputs."""
    logger.info("qwen_image_inpaint_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_inpaint", "Qwen-Image")
        url = "http://127.0.0.1:6970/qwen_image_inpaint_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_inpaint_nunchaku_generate", response_model=QwenImageResponse)
async def qwen_image_inpaint_nunchaku_generate(data: QwenImageInpaintRequest = Body(...)):
    """Generates some number of qwen image inpaint images based on user inputs."""
    logger.info("qwen_image_inpaint_nunchaku_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_inpaint_nunchaku", "Qwen-Image")
        url = "http://127.0.0.1:6970/qwen_image_inpaint_nunchaku_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}


@avernus.post("/qwen_image_edit_generate", response_model=QwenImageResponse)
async def qwen_image_edit_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_edit", "Qwen-Image-Edit")
        url = "http://127.0.0.1:6970/qwen_image_edit_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_edit_nunchaku_generate", response_model=QwenImageResponse)
async def qwen_image_edit_nunchaku_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_nunchaku_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_edit_nunchaku", "Qwen-Image-Edit")
        url = "http://127.0.0.1:6970/qwen_image_edit_nunchaku_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}


@avernus.post("/qwen_image_edit_plus_generate", response_model=QwenImageResponse)
async def qwen_image_edit_plus_generate(data: QwenImageEditPlusRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_edit_plus", "Qwen-Image-Edit-Plus")
        url = "http://127.0.0.1:6970/qwen_image_edit_plus_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/qwen_image_edit_plus_nunchaku_generate", response_model=QwenImageResponse)
async def qwen_image_edit_plus_nunchaku_generate(data: QwenImageEditPlusRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("qwen_image_edit_plus_nunchaku", "Qwen-Image-Edit-Plus")
        url = "http://127.0.0.1:6970/qwen_image_edit_plus_nunchaku_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/realesrgan_generate", response_model=RealESRGANResponse)
async def realesrgan_generate(data: RealESRGANRequest = Body(...)):
    """Upscales an image based on user input"""
    logger.info("realesrgan request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("realesrgan", data.scale)
        url = "http://127.0.0.1:6970/realesrgan_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.get("/status", response_model=StatusResponse)
async def status():
    """ This returns Ok when hit"""
    return {"status": str("Ok!"),
            "version": str("0.6.0")}

@avernus.post("/sdxl_generate", response_model=SDXLResponse)
async def sdxl_generate(data: SDXLRequest = Body(...)):
    """Generates some number of sdxl images based on user inputs."""
    logger.info("sdxl_generate request received")
    async with pipeline_lock:
        if data.image is None and data.controlnet_image is None:
            await server_manager.set_pipeline("sdxl", data.model_name)
        if data.image is None and data.controlnet_image is not None:
            await server_manager.set_pipeline("sdxl_controlnet", data.model_name)
        if data.image is not None and data.controlnet_image is None:
            await server_manager.set_pipeline("sdxl_i2i", data.model_name)
        if data.image is not None and data.controlnet_image is not None:
            await server_manager.set_pipeline("sdxl_i2i_controlnet", data.model_name)
        url = "http://127.0.0.1:6970/sdxl_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/sdxl_inpaint_generate", response_model=SDXLResponse)
async def sdxl_inpaint_generate(data: SDXLInpaintRequest = Body(...)):
    """Generates some number of sdxl inpaint images based on user inputs."""
    logger.info("sdxl_inpaint_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("sdxl_inpaint", data.model_name)
        url = "http://127.0.0.1:6970/sdxl_inpaint_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/swin2sr_generate", response_model=Swin2SRResponse)
async def swin2sr_generate(data: Swin2SRRequest = Body(...)):
    """Upscales an image based on user input"""
    logger.info("swin2sr request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("swin2sr")
        url = "http://127.0.0.1:6970/swin2sr_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return result
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/wan_ti2v_generate")
async def wan_ti2v_generate(data: WanTI2VRequest = Body(...)):
    logger.info("wan_ti2v_generate request received")
    async with pipeline_lock:
        if data.image is None:
            await server_manager.set_pipeline("wan", data.model_name)
        if data.image is not None:
            await server_manager.set_pipeline("wan_i2v", data.model_name)
        url = "http://127.0.0.1:6970/wan_ti2v_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return StreamingResponse(cleanup_and_stream(result["path"]), media_type="video/mp4")
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/wan_vace_generate")
async def wan_vace_generate(data: WanVACERequest = Body(...)):
    logger.info("wan_vace_generate request received")
    async with pipeline_lock:
        await server_manager.set_pipeline("wan_vace", data.model_name)
        url = "http://127.0.0.1:6970/wan_vace_generate"
        try:
            result = await forward_post_request(url, data)
            if result["status"] is True:
                return StreamingResponse(cleanup_and_stream(result["path"]), media_type="video/mp4")
            else:
                logger.error(f"Generation Error: {result['status_message']}")
                server_manager.kill_pipeline()
                return {"status": False,
                        "status_message": result["status_message"]}
        except Exception as e:
            server_manager.kill_pipeline()
            return {"status": False,
                    "status_message": str(e)}

@avernus.post("/wan_v2v_generate")
async def wan_v2v_generate(
    prompt: str = Form(...),
    negative_prompt: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    flow_shift: float | None = Form(None),
    seed: int | None = Form(None),
    model_name: str | None = Form(None),
    video: UploadFile | None = File(None)  # 👈 NEW
):
    logger.info("wan_v2v_generate request received")

    async with pipeline_lock:
        await server_manager.set_pipeline("wan_v2v", model_name)
        url = "http://127.0.0.1:6970/wan_v2v_generate"
        form_data = {"prompt": prompt,
                     "negative_prompt": negative_prompt,
                     "width": width,
                     "height": height,
                     "steps": steps,
                     "guidance_scale": guidance_scale,
                     "flow_shift": flow_shift,
                     "seed": seed,
                     "model_name": model_name}
        form_data = {k: str(v) for k, v in form_data.items() if v is not None}
        files = {}
        if video:
            video_bytes = await video.read()
            files["video"] = (video.filename, video_bytes, video.content_type)
        else:
            return JSONResponse({"status": False, "status_message": "Video required"}, status_code=500)
        try:
            async with httpx.AsyncClient(timeout=360000) as client:
                result = await client.post(url, data=form_data, files=files)
            result_json = result.json()

            if result_json["status"] is True:
                return StreamingResponse(cleanup_and_stream(result_json["path"]), media_type="video/mp4")
            else:
                logger.error(f"Generation Error: {result_json['status_message']}")
                server_manager.kill_pipeline()
                return JSONResponse(result_json, status_code=500)
        except Exception as e:
            server_manager.kill_pipeline()
            return JSONResponse({"status": False, "status_message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus, host="0.0.0.0")