import gc
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel, T5TokenizerFast




async def generate_ltx(prompt, video=None):
    prompt_embeds, prompt_attention_mask = await get_t5_prompt_embeds(prompt)
    negative_prompt_embeds, negative_prompt_attention_mask = await get_t5_prompt_embeds("")

    transformer = LTXVideoTransformer3DModel.from_pretrained("Lightricks/LTX-Video",
                                                             subfolder="transformer",
                                                             torch_dtype=torch.float16)

    generator = LTXPipeline.from_pretrained("Lightricks/LTX-Video",
                                            transformer=transformer,
                                            torch_dtype=torch.float16)

    video = generator(prompt_embeds=prompt_embeds,
                      prompt_attention_mask=prompt_attention_mask,
                      negative_prompt_embeds=negative_prompt_embeds,
                      negative_prompt_attention_mask=negative_prompt_attention_mask,
                      width=704,
                      height=480,
                      num_frames=161,
                      num_inference_steps=50).frames[0]


    del transformer, generator
    torch.cuda.empty_cache()
    gc.collect()

    return video

async def get_t5_prompt_embeds(prompt,
                               num_videos_per_prompt: int = 1,
                               max_sequence_length: int = 128,
                               ):
    model_name = "Lightricks/LTX-Video"

    tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer")

    prompt = [prompt] if isinstance(prompt, str) else prompt

    batch_size = len(prompt)

    prompt_text_inputs = tokenizer(prompt,
                            padding="max_length",
                            max_length=max_sequence_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt",
    )
    text_input_ids = prompt_text_inputs.input_ids
    prompt_attention_mask = prompt_text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool()

    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=torch.float16)


    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)


    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    return prompt_embeds, prompt_attention_mask
