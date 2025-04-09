import base64
from io import BytesIO
import os
from fastapi import FastAPI, Request, Body
from modules.logging_config import setup_logging
from modules.chat import generate_chat, generate_multimodal_chat
from modules.sdxl import generate_sdxl
from modules.flux import generate_flux
from modules.pydantic_models import (FluxRequest, FluxResponse, FluxLoraListResponse, SDXLRequest, SDXLResponse,
                                     SDXLLoraListResponse, LLMRequest, LLMResponse, MultiModalLLMRequest,
                                     MultiModalLLMResponse, StatusResponse)
from loguru import logger
from PIL import Image

setup_logging()
avernus = FastAPI()

@avernus.get("/status", response_model=StatusResponse)
async def status():
    """ This returns Ok when hit"""
    logger.info("status request received")
    return {"status": str("Ok!")}

@avernus.post("/llm_chat", response_model=LLMResponse)
async def llm_chat(request: Request, data: LLMRequest = Body(...)):
    """This takes a prompt, and optionally a Huggingface model name, and/or a Huggingface formatted message history.
    See test_harness.py for an example of one"""
    logger.info(f"{request.client.host}:{request.client.port} - llm_chat request received")
    kwargs = {"prompt": data.prompt,
              "model_name": data.model_name,
              "messages": data.messages}
    try:
        response = await generate_chat(**kwargs)
    except Exception as e:
        logger.error(f"llm_chat ERROR: {e}")
        return {"error": str(e)}
    return {"response": response}

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
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.strength is not None and data.image:
        kwargs["strength"] = data.strength
    try:
        response = await generate_sdxl(**kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"sdxl_generate ERROR: {e}")
        return None
    return {"images": base64_images}

@avernus.post("/flux_generate", response_model=FluxResponse)
async def flux_generate(data: FluxRequest = Body(...)):
    """Generates some number of Flux images based on user inputs"""
    logger.info("flux_generate request received")
    kwargs = {"prompt": data.prompt,
              "width": data.width,
              "height": data.height,
              "steps": data.steps,
              "batch_size": data.batch_size,
              "model_name": data.model_name}
    if data.lora_name:
        kwargs["lora_name"] = data.lora_name
    if data.image:
        kwargs["image"] = base64_to_image(data.image)
    if data.strength is not None and data.image:
        kwargs["strength"] = data.strength
    try:
        response = await generate_flux(**kwargs)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_generate ERROR: {e}")
        return None
    return {"images": base64_images}

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
