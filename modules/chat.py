import torch
from transformers import pipeline
from transformers import logging as t_logging
import gc
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
t_logging.disable_progress_bar()

async def generate_chat(prompt, model_name, messages=None):
    if model_name is None:
        model_name ="Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4"
    generator = pipeline("text-generation",
                         model=model_name,
                         model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}})
    if messages:
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [{"role": "user", "content": prompt}]

    outputs = generator(messages, max_new_tokens=2048, do_sample=True)
    response = outputs[0]["generated_text"][-1]["content"]
    generator.model.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return response

async def generate_multimodal_chat(prompt, model_name, image=None, messages=None):
    if model_name is None:
        model_name ="mlabonne/gemma-3-12b-it-abliterated"
    generator = pipeline("image-text-to-text",
                         model=model_name,
                         model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}})
    if messages:
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    else:
        messages = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    try:
        outputs = generator(messages, max_new_tokens=2048, do_sample=True)
    except Exception as e:
        print(f"generate_multimodal_chat ERROR: {e}")

    response = outputs[0]["generated_text"][-1]["content"]
    generator.model.to("cpu")
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    return response







