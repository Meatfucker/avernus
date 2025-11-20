from fastapi import FastAPI, Body
import torch
from transformers import pipeline

from pydantic_models import LLMRequest, LLMResponse

PIPELINE = None
LOADED: bool = False

avernus_llm = FastAPI()

def load_pipeline(model_name=None):
    global PIPELINE
    if model_name is None:
        model_name = "Meatfucker/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4-bnb-nf4"
    try:
        PIPELINE = pipeline(task="text-generation",
                            model=model_name,
                            model_kwargs={"torch_dtype": torch.bfloat16})
    except Exception as e:
        print(f"LOADING EXCEPTION: {e}")

def generate_chat(prompt, model_name, messages=None):
    global PIPELINE
    global LOADED
    if not LOADED:
        load_pipeline(model_name)
        LOADED = True
    if messages:
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [{"role": "user", "content": prompt}]
    try:
        outputs = PIPELINE(messages, max_new_tokens=2048, do_sample=True)
        response = outputs[0]["generated_text"][-1]["content"]
        return {"status": True,
                "response": response}
    except Exception as e:
        return {"status": False,
                "status_message": f"FUNCTION ERROR: {str(e)}"}

@avernus_llm.post("/llm_chat", response_model=LLMResponse)
def llm_chat(data: LLMRequest = Body(...)):
    """This takes a prompt, and optionally a Huggingface model name, and/or a Huggingface formatted message history.
    See test_harness.py for an example of one"""
    kwargs = {"prompt": data.prompt,
              "model_name": data.model_name,
              "messages": data.messages}
    try:
        response = generate_chat(**kwargs)
        if response["status"] is True:
            return response
        else:
            return {"status": False,
                    "status_message": str(response["status_message"])}
    except Exception as e:
        return {"status": False,
                "status_message": f"API ERROR: {str(e)}, KWARGS:{kwargs}"}

@avernus_llm.get("/online")
async def status():
    """ This returns True when hit"""
    return True

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(avernus_llm, host="0.0.0.0", port=6970, log_level="critical")