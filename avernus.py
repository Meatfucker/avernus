import base64
import gc
from io import BytesIO
import os
from typing import Optional
from diffusers.utils import export_to_video
from fastapi import FastAPI, Request, Body, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from modules.pydantic_models import (ACEStepRequest, FluxInpaintRequest, FluxRequest, FluxResponse,
                                     FluxLoraListResponse, LLMRequest, LLMResponse, MultiModalLLMRequest,
                                     MultiModalLLMResponse, QwenImageRequest, QwenImageInpaintRequest,
                                     QwenImageLoraListResponse, QwenImageResponse, RAGResponse, RAGRequest,
                                     SDXLInpaintRequest, SDXLRequest, SDXLResponse, SDXLLoraListResponse,
                                     SDXLControlnetListResponse, StatusResponse, SDXLSchedulerListResponse,
                                     WanTI2VRequest)

from modules.ace import generate_ace
from modules.chat import generate_chat, generate_multimodal_chat
from modules.flux import generate_flux, generate_flux_inpaint, generate_flux_fill, generate_flux_kontext
from modules.logging_config import setup_logging
from modules.ltx import generate_ltx
from modules.qwen_image import generate_qwen_image, generate_qwen_image_inpaint, generate_qwen_image_edit
from modules.rag import retrieve_rag
from modules.sdxl import generate_sdxl, generate_sdxl_inpaint
from modules.wan import generate_wan_ti2v


from loguru import logger
from PIL import Image
import torch

setup_logging()
avernus = FastAPI()


class CurrentPipeline:
    def __init__(self,
                 pipeline=None,
                 model_type=None,
                 model_name=None):
        self.pipeline = pipeline
        self.model_type = model_type
        self.model_name = model_name

    async def delete_pipeline(self):
        self.pipeline = None
        self.model_type = None
        self.model_name = None
        torch.cuda.empty_cache()
        gc.collect()

    async def set_pipeline(self, pipeline=None, model_type=None, model_name=None):
        self.pipeline = pipeline
        self.model_type = model_type
        self.model_name = model_name

PIPELINE = CurrentPipeline()


@avernus.post("/ace_generate")
async def ace_generate(data: ACEStepRequest = Body(...)):
    """Generates audio based on user inputs"""
    logger.info("ace_generate request received")
    kwargs = {"prompt": data.prompt,
              "lyrics": data.lyrics}
    if data.actual_seeds:
        kwargs["actual_seeds"] = data.actual_seeds
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.audio_duration:
        kwargs["audio_duration"] = data.audio_duration
    if data.infer_step:
        kwargs["infer_step"] = data.infer_step
    if data.omega_scale:
        kwargs["omega_scale"] = data.omega_scale

    try:
        response = await generate_ace(PIPELINE, **kwargs)
    except Exception as e:
        logger.info(f"ace_generate ERROR: {e}")

    return StreamingResponse(open(response, "rb"), media_type="audio/wav")

@avernus.post("/flux_generate", response_model=FluxResponse)
async def flux_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
    logger.info("flux_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_flux(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/flux_inpaint_generate", response_model=FluxResponse)
async def flux_inpaint_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux inpaint images based on user inputs."""
    logger.info("flux_inpaint_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if data.strength:
        kwargs["strength"] = data.strength
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.mask_image:
        kwargs["mask_image"] = base64_to_image(data.mask_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_flux_inpaint(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_inpaint_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/flux_fill_generate", response_model=FluxResponse)
async def flux_fill_generate(data: FluxInpaintRequest = Body(...)):
    """Generates some number of flux fill images based on user inputs."""
    logger.info("flux_fill_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if data.strength:
        kwargs["strength"] = data.strength
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.mask_image:
        kwargs["mask_image"] = base64_to_image(data.mask_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_flux_fill(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_fill_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/flux_kontext_generate", response_model=FluxResponse)
async def flux_kontext_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux Kontext images based on user inputs"""
    logger.info("flux_kontext_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_flux_kontext(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_kontext_generate ERROR: {e}")
        return None
    return {"images": base64_images}


@avernus.get("/list_flux_loras", response_model=FluxLoraListResponse)
async def list_flux_loras():
    """Returns a list of the files located in the flux loras directory"""
    logger.info("list_flux_loras request received")
    try:
        loras_dir = "loras/flux"
        if not os.path.exists(loras_dir):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
        return {"loras": filenames}
    except Exception as e:
        logger.error(f"list_flux_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_qwen_image_loras", response_model=QwenImageLoraListResponse)
async def list_qwen_image_loras():
    """Returns a list of the files located in the qwen_image loras directory"""
    logger.info("list_qwen_image_loras request received")
    try:
        loras_dir = "loras/qwen_image"
        if not os.path.exists(loras_dir):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
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
        loras_dir = "loras/sdxl"
        if not os.path.exists(loras_dir):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
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
async def llm_chat(request: Request, data: LLMRequest = Body(...)):
    """This takes a prompt, and optionally a Huggingface model name, and/or a Huggingface formatted message history.
    See test_harness.py for an example of one"""
    logger.info(f"{request.client.host}:{request.client.port} - llm_chat request received")
    kwargs = {"prompt": data.prompt,
              "model_name": data.model_name,
              "messages": data.messages}
    try:
        response = await generate_chat(PIPELINE, **kwargs)
    except Exception as e:
        logger.error(f"llm_chat ERROR: {e}")
        return {"error": str(e)}
    return {"response": response}

@avernus.post("/ltx_generate")
async def ltx_generate(prompt: str = Form(...),
                       input_video: Optional[UploadFile] = File(None),
                       height: Optional[int] = None,
                       width: Optional[int] = None,
                       seed: Optional[int] = None,
                       guidance_scale: Optional[float] = None,
                       num_frames: Optional[int] = None
                       ):
    logger.info("ltx_generate request received")
    kwargs = {"prompt": prompt}
    if input_video:
        video_bytes = await input_video.read()
        kwargs["input_video"] = video_bytes
    if height:
        kwargs["height"] = height
    if width:
        kwargs["width"] = width
    if seed:
        kwargs["seed"] = seed
    if guidance_scale:
        kwargs["guidance_scale"] = guidance_scale
    if num_frames:
        kwargs["num_frames"] = num_frames

    generated_video = await generate_ltx(**kwargs)
    export_to_video(generated_video, "output.mp4", fps=24)

    return StreamingResponse(open("output.mp4", "rb"), media_type="input_video/mp4")

@avernus.post("/multimodal_llm_chat", response_model=MultiModalLLMResponse)
async def multimodal_llm_chat(data: MultiModalLLMRequest = Body(...)):
    """Same as above but also will eventually take a base64 image for multimodal chat. Does not currently work"""
    logger.info("multimodal_llm_chat request received")
    try:
        prompt = data.prompt
        model_name = data.model_name
        messages = data.messages
        response = await generate_multimodal_chat(prompt, model_name, messages=messages)
    except Exception as e:
        logger.error(f"multimodal_llm_chat ERROR: {e}")
        return {"error": str(e)}
    return response

@avernus.post("/qwen_image_generate", response_model=QwenImageResponse)
async def qwen_image_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image images based on user inputs"""
    logger.info("qwen_image_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
        kwargs["strength"] = data.strength
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_qwen_image(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"qwen_image_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/qwen_image_inpaint_generate", response_model=QwenImageResponse)
async def qwen_image_inpaint_generate(data: QwenImageInpaintRequest = Body(...)):
    """Generates some number of qwen image inpaint images based on user inputs."""
    logger.info("qwen_image_inpaint_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.strength:
        kwargs["strength"] = data.strength
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.mask_image:
        kwargs["mask_image"] = base64_to_image(data.mask_image)
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_qwen_image_inpaint(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"qwen_image_inpaint_generate ERROR: {e}")
        return None
    return {"images": base64_images}


@avernus.post("/qwen_image_edit_generate", response_model=QwenImageResponse)
async def qwen_image_edit_generate(data: QwenImageRequest = Body(...)):
    """Generates some number of Qwen Image Edit images based on user inputs"""
    logger.info("qwen_image_edit_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.true_cfg_scale:
        kwargs["true_cfg_scale"] = data.true_cfg_scale
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_qwen_image_edit(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"qwen_image_edit_generate ERROR: {e}")
        return None
    return {"images": base64_images}


@avernus.get("/status", response_model=StatusResponse)
async def status():
    """ This returns Ok when hit"""
    logger.info("status request received")
    return {"status": str("Ok!"), "version": str("0.6.0")}

@avernus.post("/rag_retrieve", response_model=RAGResponse)
async def rag_retrieve(request: Request, data: RAGRequest = Body(...)):
    """This takes a prompt and an optional number of results to return and then returns a list of strings"""
    logger.info(f"{request.client.host}:{request.client.port} - rag request received")
    kwargs = {"prompt": data.prompt}
    if data.max_candidates:
        kwargs["max_candidates"] = data.max_candidates
    if data.similarity_threshold:
        kwargs["similarity_threshold"] = data.similarity_threshold
    try:
        response = await retrieve_rag(**kwargs)
    except Exception as e:
        logger.error(f"rag_retrieve ERROR: {e}")
        return {"error": str(e)}
    return {"response": response}

@avernus.post("/sdxl_generate", response_model=SDXLResponse)
async def sdxl_generate(data: SDXLRequest = Body(...)):
    """Generates some number of sdxl images based on user inputs."""
    logger.info("sdxl_generate request received")
    kwargs = {"prompt": data.prompt,
              "negative_prompt": data.negative_prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size,
              "model_name": data.model_name}
    if data.controlnet_processor:
        kwargs["controlnet_processor"] = data.controlnet_processor
        kwargs["controlnet_image"] = base64_to_image(data.controlnet_image)
    if data.controlnet_conditioning:
        kwargs["controlnet_conditioning"] = data.controlnet_conditioning
    if data.ip_adapter_image:
        kwargs["ip_adapter_strength"] = data.ip_adapter_strength
        kwargs["ip_adapter_image"] = base64_to_image(data.ip_adapter_image)
    if data.scheduler:
        kwargs["scheduler"] = data.scheduler
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.strength is not None and data.image:
        kwargs["strength"] = data.strength
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    try:
        response = await generate_sdxl(PIPELINE, **kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"sdxl_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/sdxl_inpaint_generate", response_model=SDXLResponse)
async def sdxl_inpaint_generate(data: SDXLInpaintRequest = Body(...)):
    """Generates some number of sdxl inpaint images based on user inputs."""
    logger.info("sdxl_inpaint_generate request received")
    kwargs = {"prompt": data.prompt,
              "negative_prompt": data.negative_prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size,
              "model_name": data.model_name}
    if data.scheduler:
        kwargs["scheduler"] = data.scheduler
    if data.strength:
        kwargs["strength"] = data.strength
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.mask_image:
        kwargs["mask_image"] = base64_to_image(data.mask_image)
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if isinstance(data.lora_name, str):
        kwargs["lora_name"] = [data.lora_name]
    else:
        kwargs["lora_name"] = data.lora_name
    if data.seed:
        kwargs["seed"] = data.seed

    try:
        response = await generate_sdxl_inpaint(**kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"sdxl_inpaint_generate ERROR: {e}")
        return None
    return {"images": base64_images}

#@avernus.post("/wan_generate")
#async def wan_generate(prompt: str = Form(...),
#                       negative_prompt: Optional[str] = None,
#                       input_video: Optional[UploadFile] = File(None),
#                       height: Optional[int] = None,
#                       width: Optional[int] = None,
#                       seed: Optional[int] = None,
#                       guidance_scale: Optional[float] = None,
#                       num_frames: Optional[int] = None
#                       ):
#    logger.info("wan_generate request received")
#    kwargs = {"prompt": prompt}
#    if negative_prompt:
#        kwargs["negative_prompt"] = negative_prompt
#    if input_video:
#        video_bytes = await input_video.read()
#        kwargs["input_video"] = video_bytes
#    if height:
#       kwargs["height"] = height
#    if width:
#        kwargs["width"] = width
#    if seed:
#        kwargs["seed"] = seed
#    if guidance_scale:
#        kwargs["guidance_scale"] = guidance_scale
#    if num_frames:
#        kwargs["num_frames"] = num_frames

#    generated_video = await generate_wan(**kwargs)
#    export_to_video(generated_video, "wan_output.mp4", fps=24)
#    return StreamingResponse(open("wan_output.mp4", "rb"), media_type="input_video/mp4")
#    return None

@avernus.post("/wan_ti2v_generate")
async def wan_ti2v_generate(data: WanTI2VRequest = Body(...)):
    logger.info("wan_ti2v_generate request received")
    kwargs = {"prompt": data.prompt}
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.width:
        kwargs["width"] = data.width
    if data.height:
        kwargs["height"] = data.height
    if data.steps:
        kwargs["steps"] = data.steps
    if data.num_frames:
        kwargs["num_frames"] = data.num_frames
    if data.guidance_scale:
        kwargs["guidance_scale"] = data.guidance_scale
    if data.seed:
        kwargs["seed"] = data.seed
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.model_name:
        kwargs["model_name"] = data.model_name

    generated_video = await generate_wan_ti2v(PIPELINE, **kwargs)
    export_to_video(generated_video, "wan_output.mp4", fps=24)

    return StreamingResponse(open("wan_output.mp4", "rb"), media_type="input_video/mp4")

def image_to_base64(image):
    """Takes a PIL image and converts it to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string):
    """Takes a base64 image and converts it to a PIL image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus, host="0.0.0.0")
