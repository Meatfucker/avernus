from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class StatusResponse(BaseModel):
    status: str = Field(..., example="Ok!")
    version: str = Field(..., example="0.2.0")

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

class FluxRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    model_name: Optional[str] = Field(None, example="Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[str] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)

class FluxResponse(BaseModel):
    images: List[str] = Field(..., example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class FluxLoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])

class SDXLRequest(BaseModel):
    prompt: str = Field(..., example="A big green monster")
    negative_prompt: Optional[str] = Field(None, example="a blue dog")
    model_name: Optional[str] = Field(None, example="Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4")
    width: Optional[int] = Field(None, example=1024)
    height: Optional[int] = Field(None, example=1024)
    steps: Optional[int] = Field(None, example=30)
    batch_size: Optional[int] = Field(None, example=4)
    lora_name: Optional[str] = Field(None, example="lora_name.safetensors")
    image: Optional[str] = Field(None, example="a9d8fp0sa9dfpasdfllkajsdflkjadslf...")
    strength: Optional[float] = Field(None, example=0.75)

class SDXLResponse(BaseModel):
    images: List[str] = Field(..., example=["kajsdflsadfsadf....", "lkjdsaflkjsadlkfjsa3423....", "lwerewjrlwkejrwewr..."])

class SDXLLoraListResponse(BaseModel):
    loras: List[str] = Field(..., example=["lora1.safetensors", "lora2.safetensors", "lora3.safetensors"])