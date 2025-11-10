from typing import Any

from fastapi import FastAPI, Body
import torch

from pydantic_models import RealESRGANRequest, ImageResponse
from RealESRGAN import RealESRGAN
from utils import base64_to_image, image_to_base64

PIPELINE: RealESRGAN
LOADED: bool = False
avernus_realesrgan = FastAPI()


def load_realesrgan(scale=4):
    global PIPELINE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        PIPELINE = RealESRGAN(device, scale=scale)
        PIPELINE.load_weights(f"./models/RealESRGAN_x{scale}.pth", download=True)
    except Exception as e:
        print(e)

def generate_realesrgan_image(image, scale=4):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_realesrgan(scale)
        LOADED = True

    try:
        upscaled_image = PIPELINE.predict(image)
        return {"status": True,
                "image": upscaled_image}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_realesrgan.post("/realesrgan_generate", response_model=ImageResponse)
def realesrgan_generate(data: RealESRGANRequest = Body(...)):
    """Generates an upscaled image using RealESRGAN"""
    kwargs: dict[str, Any] = {"image": base64_to_image(data.image),
                              "scale": data.scale}
    try:
        response = generate_realesrgan_image(**kwargs)
        if response["status"] is True:
            base64_image = image_to_base64(response["image"])
            response = None
            del response
        else:
            return {"status": False,
                    "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Real-ESRGAN Success",
            "images": [base64_image]}

@avernus_realesrgan.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(avernus_realesrgan, host="0.0.0.0", port=6970)
    #uvicorn.run(avernus_realesrgan, host="0.0.0.0", port=6970, log_level="critical")