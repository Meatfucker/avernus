from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class GenericResponse(BaseModel):
    status: bool = Field(..., example=True)
    status_message: Optional[str] = Field(None, example="Generation Success")

class ACEStepRequest(BaseModel):
    prompt: str = Field(..., example="Weird jam band music, guitar, bass")
    lyrics: str = Field(..., example="[chorus] Blah blah blah, [inst]")
    audio_duration: Optional[float] = Field(None, example=60.0)
    infer_step: Optional[int] = Field(None, example=60)
    guidance_scale: Optional[float] = Field(None, example=15.0)
    omega_scale: Optional[float] = Field(None, example=10.0)
    actual_seeds: Optional[int] = Field(None, example=42)

class AuraFlowRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="clams")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="fal/AuraFlow")

class ChromaRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="blurry")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.9)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/Flux.1-dev-bnb-nf4")

class FluxControlnetListResponse(BaseModel):
    flux_controlnets: List[str] = Field(..., example=["canny", "depth"])

class FluxInpaintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="clams")
    model_name: Optional[str] = Field(None, example="some-repo/some-flux-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    mask_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    true_cfg_scale: Optional[float] = Field(None, example=1.0)

class FluxRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="clams")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    ip_adapter_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    ip_adapter_strength: Optional[float] = Field(None, example=0.6)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/Flux.1-dev-bnb-nf4")
    true_cfg_scale: Optional[float] = Field(None, example=1.0)

class FramepackRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=1280)
    height: Optional[int] = Field(None, example=720)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=129)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    last_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    guidance_scale: Optional[float] = Field(None, example=6.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/HunyuanVideo-bnb-nf4")

class HiDreamRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="clams")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/Flux.1-dev-bnb-nf4")

class HunyuanTI2VRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=832)
    height: Optional[int] = Field(None, example=480)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=81)
    flow_shift: Optional[float] = Field(None, example=3.0)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/HunyuanVideo-bnb-nf4")

class ImageResponse(GenericResponse):
    images: Optional[List[str]] = Field(None, example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class ImageGenAuxRequest(BaseModel):
    image: str = Field(..., example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    model: Optional[str] = Field(None, example="OzzyGT/DAT_X4")
    scale: Optional[int] = Field(None, example=4)
    tiling: Optional[bool] = Field(None, example=True)
    tile_width: Optional[int] = Field(None, example=768)
    tile_height: Optional[int] = Field(None, example=768)
    overlap: Optional[int] = Field(None, example=8)

class ImageGenAuxResponse(GenericResponse):
    images: Optional[str] = Field(None, example="kajsdflsadfsadf....")

class KandinskyT2VRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=832)
    height: Optional[int] = Field(None, example=480)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=81)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/HunyuanVideo-bnb-nf4")

class LLMRequest(BaseModel):
    prompt: str = Field(..., example="Who is the best at hackey sack?")
    model_name: Optional[str] = Field(None, example="Goekdeniz-Guelmez/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2")
    messages: Optional[List[Dict]] = Field(None, example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there, how may I assist you?"}])

class LLMResponse(GenericResponse):
    response: Optional[str] = Field(None, example="Hi Im online and how can I help you?")

class LoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])

class LTXTI2VRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=832)
    height: Optional[int] = Field(None, example=480)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=81)
    frame_rate: Optional[int] = Field(None, example=25)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Wan-AI/Wan2.2-TI2V-5B-Diffusers")

class LuminaRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="clams")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/Flux.1-dev-bnb-nf4")

class QwenImageInpaintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a red dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-qwen-image-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=50)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    mask_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    true_cfg_scale: Optional[float] = Field(None, example=4.0)
    seed: Optional[int] = Field(None, example=42)

class QwenImageRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a red dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-qwen-image-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=50)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    true_cfg_scale: Optional[float] = Field(None, example=1.0)
    seed: Optional[int] = Field(None, example=42)

class QwenImageEditPlusRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a red dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-qwen-image-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=50)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    images: Optional[List[str]] = Field(None, example=["a9d8fp0sa9dfpasdfllkajsdflkjadslf...", "a9d8fp0sa9dfpasdfllkajsdflkjadslf..."])
    strength: Optional[float] = Field(None, example=0.75)
    true_cfg_scale: Optional[float] = Field(None, example=1.0)
    seed: Optional[int] = Field(None, example=42)

class RAGRequest(BaseModel):
    prompt: str = Field(..., example="What is answer to life, the universe, and everything")
    max_candidates: Optional[int] = Field(..., example=20)
    similarity_threshold: Optional[float] = Field(..., example=0.8)

class RAGResponse(BaseModel):
    response: List[str] = Field(..., example=["glip", "glop", "blip", "blop"])

class RealESRGANRequest(BaseModel):
    image: str = Field(..., example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    scale: Optional[int] = Field(None, example=4)

class RealESRGANResponse(GenericResponse):
    images: Optional[str] = Field(None, example="kajsdflsadfsadf....")

class SanaSprintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    max_timesteps: Optional[float] = Field(None, example=1.57080)
    intermediate_timesteps: Optional[float] = Field(None, example=1.3)
    batch_size: Optional[int] = Field(None, example=4)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.9)
    guidance_scale: Optional[float] = Field(None, example=4.5)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Meatfucker/Flux.1-dev-bnb-nf4")

class SD15Request(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-sd-model")
    width: Optional[int] = Field(None, example=512)
    height: Optional[int] = Field(None, example=512)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    guidance_scale: Optional[float] = Field(None, example=7.5)
    scheduler: Optional[str] = Field(None, example="DPMSolverMultistepScheduler")
    seed: Optional[int] = Field(None, example=42)

class SD15InpaintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-sdxl-model")
    width: Optional[int] = Field(None, example=512)
    height: Optional[int] = Field(None, example=512)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    mask_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    guidance_scale: Optional[float] = Field(None, example=7.5)
    scheduler: Optional[str] = Field(None, example="DPMSolverMultistepScheduler")
    seed: Optional[int] = Field(None, example=42)

class SDXLControlnetListResponse(BaseModel):
    sdxl_controlnets: List[str] = Field(..., example=["canny", "depth"])

class SDXLInpaintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-sdxl-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    mask_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    scheduler: Optional[str] = Field(None, example="DPMSolverMultistepScheduler")
    seed: Optional[int] = Field(None, example=42)

class SDXLRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    model_name: Optional[str] = Field(None, example="some-repo/some-sdxl-model")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[Union[str, List[str]]] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)
    controlnet_processor: Optional[str] = Field(None, example="canny")
    controlnet_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    controlnet_conditioning: Optional[float] = Field(None, example=0.5)
    ip_adapter_image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    ip_adapter_strength: Optional[float] = Field(None, example=0.6)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    scheduler: Optional[str] = Field(None, example="DPMSolverMultistepScheduler")
    seed: Optional[int] = Field(None, example=42)

class SDXLSchedulerListResponse(BaseModel):
    schedulers: List[str] = Field(..., example=["DPMSolverMultistepScheduler", "EulerAncestralDiscreteScheduler", "LMSDiscreteScheduler"])

class StatusResponse(BaseModel):
    status: str = Field(..., example="Ok!")
    version: str = Field(..., example="0.2.0")

class Swin2SRRequest(BaseModel):
    image: str = Field(..., example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")

class Swin2SRResponse(GenericResponse):
    images: Optional[str] = Field(None, example="kajsdflsadfsadf....")

class WanTI2VRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=832)
    height: Optional[int] = Field(None, example=480)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=81)
    flow_shift: Optional[float] = Field(None, example=3.0)
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Wan-AI/Wan2.2-TI2V-5B-Diffusers")

class WanVACERequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    width: Optional[int] = Field(None, example=832)
    height: Optional[int] = Field(None, example=480)
    steps: Optional[int] = Field(None, example=50)
    num_frames: Optional[int] = Field(None, example=81)
    first_frame: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    last_frame: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    flow_shift: Optional[float] = Field(None, example=3.0)
    guidance_scale: Optional[float] = Field(None, example=5.0)
    seed: Optional[int] = Field(None, example=42)
    model_name: Optional[str] = Field(None, example="Wan-AI/Wan2.2-TI2V-5B-Diffusers")