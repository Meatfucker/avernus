import base64
from io import BytesIO
import os
from fastapi import FastAPI, Request
from modules.chat import generate_chat, generate_multimodal_chat
from modules.sdxl import generate_sdxl, generate_lora_sdxl, generate_sdxl_i2i, generate_lora_sdxl_i2i
from modules.flux import generate_flux, generate_lora_flux, generate_flux_i2i, generate_lora_flux_i2i
from loguru import logger
from PIL import Image

avernus = FastAPI()

@avernus.get("/status")
async def status():
    """ This returns Ok when hit"""
    logger.info("status request received")
    return {"status": str("Ok!")}

@avernus.post("/llm_chat")
async def llm_chat(request: Request):
    """This takes a prompt, and optionally a Huggingface model name, and/or a Huggingface formatted message history.
    See test_harness.py for an example of one"""
    logger.info("llm_chat request received")
    try:
        data = await request.json()
        prompt = data.get("prompt")
        model_name = data.get("model_name")
        messages = data.get("messages")
        response = await generate_chat(prompt, model_name, messages)
    except Exception as e:
        logger.error(f"llm_chat ERROR: {e}")
        return {"error": str(e)}
    return response

@avernus.post("/multimodal_llm_chat")
async def multimodal_llm_chat(request: Request):
    """Same as above but also will eventually take a base64 image for multimodal chat. Does not currently work"""
    logger.info("multimodal_llm_chat request received")
    try:
        data = await request.json()
        prompt = data.get("prompt")
        model_name = data.get("model_name")
        messages = data.get("messages")
        response = await generate_multimodal_chat(prompt, model_name, messages=messages)
    except Exception as e:
        logger.error(f"multimodal_llm_chat ERROR: {e}")
        return {"error": str(e)}
    return response

@avernus.post("/sdxl_generate")
async def sdxl_generate(request: Request):
    """Generates some number of sdxl images based on user inputs."""
    logger.info("sdxl_generate request received")
    try:
        data = await request.json()
        prompt = data.get("prompt")
        negative_prompt = data.get("negative_prompt")
        model_name = data.get("model_name")
        lora_name = data.get("lora_name")
        width = data.get("width")
        height = data.get("height")
        steps = data.get("steps")
        batch_size = data.get("batch_size")
        strength = data.get("strength")
        image = data.get("image")
        if lora_name:
            if image:
                image = base64_to_image(image)
                response = await generate_lora_sdxl_i2i(prompt, image, width, height, steps, batch_size, strength,
                                                        negative_prompt=negative_prompt, model_name=model_name,
                                                        lora_name=lora_name)
            else:
                response = await generate_lora_sdxl(prompt, width, height, steps, batch_size,
                                                    negative_prompt=negative_prompt, model_name=model_name,
                                                    lora_name=lora_name)
        else:
            if image:
                image = base64_to_image(image)
                response = await generate_sdxl_i2i(prompt, image, width, height, steps, batch_size, strength,
                                                   negative_prompt=negative_prompt, model_name=model_name)
            else:
                response = await generate_sdxl(prompt, width, height, steps, batch_size,
                                               negative_prompt=negative_prompt, model_name=model_name)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"sdxl_generate ERROR: {e}")
        return None
    return base64_images

@avernus.post("/flux_generate")
async def flux_generate(request: Request):
    """Generates some number of Flux images based on user inputs"""
    logger.info("flux_generate request received")
    try:
        data = await request.json()
        prompt = data.get("prompt")
        model_name = data.get("model_name")
        width = data.get("width")
        height = data.get("height")
        steps = data.get("steps")
        batch_size = data.get("batch_size")
        lora_name = data.get("lora_name")
        image = data.get("image")
        strength = data.get("strength")
        if lora_name:
            if image:
                image = base64_to_image(image)
                response = await generate_lora_flux_i2i(prompt, image, width, height, steps, batch_size,
                                                        model_name=model_name, lora_name=lora_name, strength=strength)
            else:
                response = await generate_lora_flux(prompt, width, height, steps, batch_size,
                                                    model_name=model_name, lora_name=lora_name)
        else:
            if image:
                image = base64_to_image(image)
                response = await generate_flux_i2i(prompt, image, width, height, steps, batch_size,
                                                   model_name=model_name, strength=strength)
            else:
                response = await generate_flux(prompt, width, height, steps, batch_size, model_name=model_name)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_generate ERROR: {e}")
        return None
    return base64_images

@avernus.get("/list_sdxl_loras")
async def list_sdxl_loras():
    """Returns a list of the files located in the sdxl loras directory."""
    logger.info("list_sdxl_loras request received")
    try:
        loras_dir = "loras/sdxl"
        if not os.path.exists(loras_dir):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
        return filenames
    except Exception as e:
        logger.error(f"list_sdxl_loras ERROR: {e}")
        return {"error": str(e)}

@avernus.get("/list_flux_loras")
async def list_flux_loras():
    """Returns a list of the files located in the flux loras directory"""
    logger.info("list_flux_loras request received")
    try:
        loras_dir = "loras/flux"
        if not os.path.exists(loras_dir):
            return {"error": "Directory not found"}
        filenames = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
        return filenames
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
