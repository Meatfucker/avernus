import base64
from io import BytesIO
import os
from fastapi import FastAPI, Request
from modules.chat import generate_chat
from modules.sdxl import generate_sdxl, generate_lora_sdxl
from modules.flux import generate_flux, generate_lora_flux
from loguru import logger

avernus = FastAPI()

@avernus.get("/status")
async def status():
    logger.info("status request received")
    return {"status": str("Ok!")}

@avernus.post("/llm_chat")
async def llm_chat(request: Request):
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

@avernus.post("/sdxl_generate")
async def sdxl_generate(request: Request):
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
        if lora_name:
            response = await generate_lora_sdxl(prompt, width, height, steps, batch_size,
                                                negative_prompt=negative_prompt, model_name=model_name,
                                                lora_name=lora_name)
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
        #if lora_name:
        #    response = await generate_lora_flux(prompt, width, height, steps, batch_size, negative_prompt=negative_prompt,
        #                                   model_name=model_name, lora_name=lora_name)
        #else:
        response = await generate_flux(prompt, width, height, steps, batch_size, model_name=model_name)
        base64_images = [image_to_base64(img) for img in response]
    except Exception as e:
        logger.info(f"flux_generate ERROR: {e}")
        return None
    return base64_images

@avernus.get("/list_sdxl_loras")
async def list_sdxl_loras():
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

@avernus.get("/list_flux_loras")
async def list_flux_loras():
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
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus, host="0.0.0.0")
