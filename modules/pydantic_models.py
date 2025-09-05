from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ACEStepRequest(BaseModel):
    prompt: str = Field(..., example="Weird jam band music, guitar, bass")
    lyrics: str = Field(..., example="[chorus] Blah blah blah, [inst]")
    audio_duration: Optional[float] = Field(None, example=60.0)
    infer_step: Optional[int] = Field(None, example=60)
    guidance_scale: Optional[float] = Field(None, example=15.0)
    omega_scale: Optional[float] = Field(None, example=10.0)
    actual_seeds: Optional[int] = Field(None, example=42)

class FluxControlnetListResponse(BaseModel):
    flux_controlnets: List[str] = Field(..., example=["canny", "depth"])

class FluxInpaintRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
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

class FluxLoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])

class FluxRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
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

class FluxResponse(BaseModel):
    images: List[str] = Field(..., example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class LLMRequest(BaseModel):
    prompt: str = Field(..., example="Who is the best at hackey sack?")
    model_name: Optional[str] = Field(None, example="Goekdeniz-Guelmez/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2")
    messages: Optional[List[Dict]] = Field(None, example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there, how may I assist you?"}])

class LLMResponse(BaseModel):
    response: str = Field(..., example="Hi Im online and how can I help you?")

class MultiModalLLMRequest(BaseModel):
    prompt: str = Field(..., example="Who is the best at hackey sack?")
    model_name: Optional[str] = Field(None, example="Goekdeniz-Guelmez/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2")
    messages: Optional[List[Dict]] = Field(None, example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there, how may I assist you?"}])
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")

class MultiModalLLMResponse(BaseModel):
    response: str = Field(..., example="Hi Im online and how can I help you?")

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

class QwenImageLoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])

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

class QwenImageResponse(BaseModel):
    images: List[str] = Field(..., example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class RAGRequest(BaseModel):
    prompt: str = Field(..., example="What is answer to life, the universe, and everything")
    max_candidates: Optional[int] = Field(..., example=20)
    similarity_threshold: Optional[float] = Field(..., example=0.8)

class RAGResponse(BaseModel):
    response: List[str] = Field(..., example=["glip", "glop", "blip", "blop"])

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

class SDXLLoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])

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

class SDXLResponse(BaseModel):
    images: List[str] = Field(..., example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class SDXLSchedulerListResponse(BaseModel):
    schedulers: List[str] = Field(..., example=["DPMSolverMultistepScheduler", "EulerAncestralDiscreteScheduler", "LMSDiscreteScheduler"])

class StatusResponse(BaseModel):
    status: str = Field(..., example="Ok!")
    version: str = Field(..., example="0.2.0")