from typing import Any

from fastapi import FastAPI, Body
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from pydantic_models import Swin2SRRequest, Swin2SRResponse
from utils import base64_to_image, image_to_base64

PIPELINE: Swin2SRForImageSuperResolution
PROCESSOR: AutoImageProcessor
LOADED: bool = False
avernus_swin2sr = FastAPI()


def load_swin2sr():
    global PIPELINE
    global PROCESSOR
    try:
        PROCESSOR = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        PIPELINE = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    except Exception as e:
        print(e)

def generate_swin2sr_image(image):
    global PIPELINE
    global PROCESSOR
    global LOADED
    if not LOADED:
        load_swin2sr()
        LOADED = True

    try:
        inputs = PROCESSOR(image, return_tensors="pt")
        with torch.no_grad():
            outputs = PIPELINE(**inputs)
        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        upscaled_image = Image.fromarray(output)
        return {"status": True,
                "image": upscaled_image}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_swin2sr.post("/swin2sr_generate", response_model=Swin2SRResponse)
def swin2sr_generate(data: Swin2SRRequest = Body(...)):
    """Generates an upscaled image using Swin2SR"""
    kwargs: dict[str, Any] = {"image": base64_to_image(data.image)}
    try:
        response = generate_swin2sr_image(**kwargs)
        if response["status"] is True:
            base64_image = image_to_base64(response["image"])
            response = None
            del response
        else:
            return {"status": False,
                    "status_message": response["status_message"]}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    return {"status": True,
            "status_message": "Swin2SR Success",
            "images": base64_image}

@avernus_swin2sr.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_swin2sr, host="0.0.0.0", port=6970)