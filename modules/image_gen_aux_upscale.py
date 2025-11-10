from typing import Any

from fastapi import FastAPI, Body
from image_gen_aux import UpscaleWithModel

from pydantic_models import ImageGenAuxRequest, ImageResponse
from utils import base64_to_image, image_to_base64

PIPELINE: UpscaleWithModel
LOADED: bool = False
avernus_image_gen_aux_upscale = FastAPI()


def load_upscaler(model="OzzyGT/DAT_X4"):
    global PIPELINE
    print("loading upscaler")
    try:
        PIPELINE = UpscaleWithModel.from_pretrained(model).to("cuda")
    except Exception as e:
        print(e)

def generate_upscaler_image(image, model="OzzyGT/DAT_X4", scale=None, tiling=None, tile_width=None, tile_height=None, overlap=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_upscaler(model)
        LOADED = True
    print("pipeline loaded")
    try:
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        if tiling is not None:
            kwargs["tiling"] = tiling
        if tile_width is not None:
            kwargs["tile_width"] = tile_width
        if tile_height is not None:
            kwargs["tile_height"] = tile_height
        if overlap is not None:
            kwargs["overlap"] = overlap
        print("generating image")
        upscaled_image = PIPELINE(image, **kwargs)
        return {"status": True,
                "image": upscaled_image}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_image_gen_aux_upscale.post("/image_gen_aux_upscale", response_model=ImageResponse)
def image_gen_aux_upscale(data: ImageGenAuxRequest = Body(...)):
    """Generates an upscaled image using image_gen_aux"""
    print("request recieved")
    try:
        kwargs: dict[str, Any] = {"image": base64_to_image(data.image),
                                  "model": data.model,
                                  "scale": data.scale,
                                  "tiling": data.tiling,
                                  "tile_width": data.tile_width,
                                  "tile_height": data.tile_height,
                                  "overlap": data.overlap}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}
    try:
        print("calling upscaler pipeline")
        response = generate_upscaler_image(**kwargs)
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
            "status_message": "image_gen_aux Upscaler Success",
            "images": [base64_image]}


@avernus_image_gen_aux_upscale.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_image_gen_aux_upscale, host="0.0.0.0", port=6970, log_level="critical")