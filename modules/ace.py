from typing import Any

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

from acestep.pipeline_ace_step import ACEStepPipeline
from pydantic_models import ACEStepRequest

PIPELINE: ACEStepPipeline
LOADED: bool = False

avernus_ace = FastAPI()


def load_ace_pipeline():
    global PIPELINE
    PIPELINE = ACEStepPipeline("./models/", dtype="bfloat16")

def generate_ace(prompt,
                 lyrics,
                 actual_seeds=None,
                 guidance_scale=None,
                 audio_duration=None,
                 infer_step=None,
                 omega_scale=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_ace_pipeline()
        LOADED = True
    kwargs = {"prompt": prompt,
              "lyrics": lyrics,
              "guidance_scale": guidance_scale if guidance_scale is not None else 15.0,
              "audio_duration": audio_duration if audio_duration is not None else 60.0,
              "infer_step": infer_step if infer_step is not None else 60,
              "omega_scale": omega_scale if omega_scale is not None else 10.0,
              "save_path": "./tests/test.wav",
              "manual_seeds": [str(actual_seeds)]}
    try:
        PIPELINE(**kwargs)
        return {"status": True,
                "path": "./tests/test.wav"}
    except Exception as e:
        return {"status": False,
                "status_message": e}

@avernus_ace.post("/ace_generate")
def ace_generate(data: ACEStepRequest = Body(...)):
    """Generates audio based on user inputs"""
    kwargs: dict[str, Any] = {"prompt": data.prompt,
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
        response = generate_ace(**kwargs)
        if response["status"] is True:
            return {"status": True,
                    "path": response["path"]}
        else:
            return {"status": False,
                    "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": str(e)}

@avernus_ace.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_ace, host="0.0.0.0", port=6970, log_level="critical")