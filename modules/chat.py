import torch
from transformers import pipeline
import gc


async def generate_chat(avernus_pipeline, prompt, model_name, messages=None):
    avernus_pipeline = await load_pipeline(avernus_pipeline, model_name)
    if messages:
        messages.append({"role": "user", "content": prompt})
    else:
        messages = [{"role": "user", "content": prompt}]

    outputs = avernus_pipeline.pipeline(messages, max_new_tokens=2048, do_sample=True)
    response = outputs[0]["generated_text"][-1]["content"]

    return response

async def load_pipeline(avernus_pipeline, model_name):
    if avernus_pipeline.model_type != "llm" or avernus_pipeline.model_name != model_name:
        await avernus_pipeline.delete_pipeline()
        if model_name is None:
            model_name = "Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4"
        generator = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}})
        await avernus_pipeline.set_pipeline(generator, "llm", model_name)

    return avernus_pipeline


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







