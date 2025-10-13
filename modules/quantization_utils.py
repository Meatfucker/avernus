from diffusers import (FluxPipeline, FluxFillPipeline, FluxKontextPipeline, FluxPriorReduxPipeline,
                       QwenImagePipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline, QwenImageTransformer2DModel,
                       AutoModel, WanImageToVideoPipeline, WanPipeline, WanVACEPipeline, HiDreamImagePipeline,
                       HiDreamImageTransformer2DModel, ChromaPipeline, WanVACETransformer3DModel)
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import (Qwen2_5_VLForConditionalGeneration, CLIPTokenizer, CLIPTextModel, T5TokenizerFast,
                          T5EncoderModel, UMT5EncoderModel, AutoModelForCausalLM, AutoTokenizer, CLIPTextModelWithProjection,
                          LlamaForCausalLM)
import torch

rank = 32
dtype = torch.bfloat16

def quantize_chroma():
    print("loading ChromaPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder"],
    )
    generator = ChromaPipeline.from_pretrained("lodestones/Chroma1-HD",
                                               quantization_config=pipeline_quant_config,
                                               torch_dtype=dtype).to("cuda")
    generator.save_pretrained("../models/Chroma1-HD")

def quantize_flux_kontext():
    print("loading FluxKontextPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                            quantization_config=pipeline_quant_config,
                                            torch_dtype=dtype).to("cuda")
    generator.save_pretrained("../models/Flux.1-Kontext-dev")

def quantize_flux_fill():
    print("loading FluxFillPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev",
                                            quantization_config=pipeline_quant_config,
                                            torch_dtype=dtype).to("cuda")
    generator.save_pretrained("../models/Flux.1-Fill-dev")

def quantize_flux_krea():
    print("loading FluxPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev",
                                            quantization_config=pipeline_quant_config,
                                            torch_dtype=dtype).to("cuda")
    generator.save_pretrained("../models/Flux.1-Krea-dev")

def quantize_flux():
    print("loading FluxPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    generator = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                            quantization_config=pipeline_quant_config,
                                            torch_dtype=dtype).to("cuda")
    generator.save_pretrained("../models/Flux.1-dev")

def quantize_hidream():
    print("loading HiDreamPipeline")
    pipeline_quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder_3"],
    )
    llama_quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_use_double_quant=True)

    tokenizer_4 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    text_encoder_4 = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                      output_hidden_states=True,
                                                      output_attentions=True,
                                                      quantization_config=llama_quant_config,
                                                      torch_dtype=torch.bfloat16).to("cpu")


    generator = HiDreamImagePipeline.from_pretrained("HiDream-ai/HiDream-I1-Full",
                                                     text_encoder_4=text_encoder_4,
                                                     tokenizer_4=tokenizer_4,
                                                     torch_dtype=dtype,
                                                     quantization_config=pipeline_quant_config)
    gen_config = generator.text_encoder_4.generation_config
    gen_config.output_attentions = False
    gen_config.output_hidden_states = False
    gen_config.return_dict_in_generate = False
    generator.save_pretrained("../models/HiDream-I1-Full")


def quantize_qwen_image():
    print("loading QwenImagePipeline")
    quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod.1",
                               "transformer_blocks.0.attn.to_q",
                               "transformer_blocks.0.attn.to_k",
                               "transformer_blocks.0.attn.to_v",
                               "transformer_blocks.0.attn.add_k_proj",
                               "transformer_blocks.0.attn.add_v_proj",
                               "transformer_blocks.0.attn.add_q_proj",
                               "transformer_blocks.0.attn.to_out.0",
                               "transformer_blocks.0.attn.to_add_out",
                               "transformer_blocks.0.img_mlp.net.0.proj",
                               "transformer_blocks.0.img_mlp.net.2",
                               "transformer_blocks.0.txt_mod.1",
                               "transformer_blocks.0.txt_mlp.net.0.proj",
                               "transformer_blocks.0.txt_mlp.net.2",
                               "transformer_blocks.59.img_mod.1",
                               "transformer_blocks.59.attn.to_q",
                               "transformer_blocks.59.attn.to_k",
                               "transformer_blocks.59.attn.to_v",
                               "transformer_blocks.59.attn.add_k_proj",
                               "transformer_blocks.59.attn.add_v_proj",
                               "transformer_blocks.59.attn.add_q_proj",
                               "transformer_blocks.59.attn.to_out.0",
                               "transformer_blocks.59.attn.to_add_out",
                               "transformer_blocks.59.img_mlp.net.0.proj",
                               "transformer_blocks.59.img_mlp.net.2",
                               "transformer_blocks.59.txt_mod.1",
                               "transformer_blocks.59.txt_mlp.net.0.proj",
                               "transformer_blocks.59.txt_mlp.net.2",
                               "norm_out.linear",
                               "proj_out"
                                ],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    transformer = transformer.to("cpu")
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_8bit=True)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="text_encoder",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    generator = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image",
                                                  transformer=transformer,
                                                  text_encoder=text_encoder,
                                                  torch_dtype=dtype)
    generator.save_pretrained("../models/Qwen-Image")

def quantize_qwen_image_edit():
    print("loading QwenImageEditPipeline")
    quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod.1",
                               "transformer_blocks.0.attn.to_q",
                               "transformer_blocks.0.attn.to_k",
                               "transformer_blocks.0.attn.to_v",
                               "transformer_blocks.0.attn.add_k_proj",
                               "transformer_blocks.0.attn.add_v_proj",
                               "transformer_blocks.0.attn.add_q_proj",
                               "transformer_blocks.0.attn.to_out.0",
                               "transformer_blocks.0.attn.to_add_out",
                               "transformer_blocks.0.img_mlp.net.0.proj",
                               "transformer_blocks.0.img_mlp.net.2",
                               "transformer_blocks.0.txt_mod.1",
                               "transformer_blocks.0.txt_mlp.net.0.proj",
                               "transformer_blocks.0.txt_mlp.net.2",
                               "transformer_blocks.59.img_mod.1",
                               "transformer_blocks.59.attn.to_q",
                               "transformer_blocks.59.attn.to_k",
                               "transformer_blocks.59.attn.to_v",
                               "transformer_blocks.59.attn.add_k_proj",
                               "transformer_blocks.59.attn.add_v_proj",
                               "transformer_blocks.59.attn.add_q_proj",
                               "transformer_blocks.59.attn.to_out.0",
                               "transformer_blocks.59.attn.to_add_out",
                               "transformer_blocks.59.img_mlp.net.0.proj",
                               "transformer_blocks.59.img_mlp.net.2",
                               "transformer_blocks.59.txt_mod.1",
                               "transformer_blocks.59.txt_mlp.net.0.proj",
                               "transformer_blocks.59.txt_mlp.net.2",
                               "norm_out.linear",
                               "proj_out"
                                ],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    transformer = transformer.to("cpu")
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules = ["transformer_blocks.0.img_mod.1",
                                 "transformer_blocks.0.attn.to_q",
                                 "transformer_blocks.0.attn.to_k",
                                 "transformer_blocks.0.attn.to_v",
                                 "transformer_blocks.0.attn.add_k_proj",
                                 "transformer_blocks.0.attn.add_v_proj",
                                 "transformer_blocks.0.attn.add_q_proj",
                                 "transformer_blocks.0.attn.to_out.0",
                                 "transformer_blocks.0.attn.to_add_out",
                                 "transformer_blocks.0.img_mlp.net.0.proj",
                                 "transformer_blocks.0.img_mlp.net.2",
                                 "transformer_blocks.0.txt_mod.1",
                                 "transformer_blocks.0.txt_mlp.net.0.proj",
                                 "transformer_blocks.0.txt_mlp.net.2",
                                 "transformer_blocks.59.img_mod.1",
                                 "transformer_blocks.59.attn.to_q",
                                 "transformer_blocks.59.attn.to_k",
                                 "transformer_blocks.59.attn.to_v",
                                 "transformer_blocks.59.attn.add_k_proj",
                                 "transformer_blocks.59.attn.add_v_proj",
                                 "transformer_blocks.59.attn.add_q_proj",
                                 "transformer_blocks.59.attn.to_out.0",
                                 "transformer_blocks.59.attn.to_add_out",
                                 "transformer_blocks.59.img_mlp.net.0.proj",
                                 "transformer_blocks.59.img_mlp.net.2",
                                 "transformer_blocks.59.txt_mod.1",
                                 "transformer_blocks.59.txt_mlp.net.0.proj",
                                 "transformer_blocks.59.txt_mlp.net.2",
                                 "norm_out.linear",
                                 "proj_out"
                                 ],
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        subfolder="text_encoder",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    generator = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit",
                                                      transformer=transformer,
                                                      text_encoder=text_encoder,
                                                      torch_dtype=dtype)
    generator.save_pretrained("../models/Qwen-Image-Edit")

def quantize_qwen_image_edit_plus():
    print("loading QwenImageEditPlusPipeline")
    quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod.1",
                               "transformer_blocks.0.attn.to_q",
                               "transformer_blocks.0.attn.to_k",
                               "transformer_blocks.0.attn.to_v",
                               "transformer_blocks.0.attn.add_k_proj",
                               "transformer_blocks.0.attn.add_v_proj",
                               "transformer_blocks.0.attn.add_q_proj",
                               "transformer_blocks.0.attn.to_out.0",
                               "transformer_blocks.0.attn.to_add_out",
                               "transformer_blocks.0.img_mlp.net.0.proj",
                               "transformer_blocks.0.img_mlp.net.2",
                               "transformer_blocks.0.txt_mod.1",
                               "transformer_blocks.0.txt_mlp.net.0.proj",
                               "transformer_blocks.0.txt_mlp.net.2",
                               "transformer_blocks.59.img_mod.1",
                               "transformer_blocks.59.attn.to_q",
                               "transformer_blocks.59.attn.to_k",
                               "transformer_blocks.59.attn.to_v",
                               "transformer_blocks.59.attn.add_k_proj",
                               "transformer_blocks.59.attn.add_v_proj",
                               "transformer_blocks.59.attn.add_q_proj",
                               "transformer_blocks.59.attn.to_out.0",
                               "transformer_blocks.59.attn.to_add_out",
                               "transformer_blocks.59.img_mlp.net.0.proj",
                               "transformer_blocks.59.img_mlp.net.2",
                               "transformer_blocks.59.txt_mod.1",
                               "transformer_blocks.59.txt_mlp.net.0.proj",
                               "transformer_blocks.59.txt_mlp.net.2",
                               "norm_out.linear",
                               "proj_out"
                                ],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    transformer = transformer.to("cpu")
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules = ["transformer_blocks.0.img_mod.1",
                                 "transformer_blocks.0.attn.to_q",
                                 "transformer_blocks.0.attn.to_k",
                                 "transformer_blocks.0.attn.to_v",
                                 "transformer_blocks.0.attn.add_k_proj",
                                 "transformer_blocks.0.attn.add_v_proj",
                                 "transformer_blocks.0.attn.add_q_proj",
                                 "transformer_blocks.0.attn.to_out.0",
                                 "transformer_blocks.0.attn.to_add_out",
                                 "transformer_blocks.0.img_mlp.net.0.proj",
                                 "transformer_blocks.0.img_mlp.net.2",
                                 "transformer_blocks.0.txt_mod.1",
                                 "transformer_blocks.0.txt_mlp.net.0.proj",
                                 "transformer_blocks.0.txt_mlp.net.2",
                                 "transformer_blocks.59.img_mod.1",
                                 "transformer_blocks.59.attn.to_q",
                                 "transformer_blocks.59.attn.to_k",
                                 "transformer_blocks.59.attn.to_v",
                                 "transformer_blocks.59.attn.add_k_proj",
                                 "transformer_blocks.59.attn.add_v_proj",
                                 "transformer_blocks.59.attn.add_q_proj",
                                 "transformer_blocks.59.attn.to_out.0",
                                 "transformer_blocks.59.attn.to_add_out",
                                 "transformer_blocks.59.img_mlp.net.0.proj",
                                 "transformer_blocks.59.img_mlp.net.2",
                                 "transformer_blocks.59.txt_mod.1",
                                 "transformer_blocks.59.txt_mlp.net.0.proj",
                                 "transformer_blocks.59.txt_mlp.net.2",
                                 "norm_out.linear",
                                 "proj_out"
                                 ],
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        subfolder="text_encoder",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    generator = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509",
                                                          transformer=transformer,
                                                          text_encoder=text_encoder,
                                                          torch_dtype=dtype)
    generator.save_pretrained("../models/Qwen-Image-Edit-Plus")

def quantize_wan21_t2v_14b():
    model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers"
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["time_embedder",
                               "timesteps_proj",
                               "time_proj",
                               "norm_out",
                               "proj_out",
                               "blocks.0.attn1.norm_k",
                               "blocks.0.attn1.norm_q",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_out.0",
                               "blocks.0.attn1.to_q",
                               "blocks.0.attn1.to_v",
                               "blocks.0.attn2.norm_k",
                               "blocks.0.attn2.norm_q",
                               "blocks.0.attn2.to_k",
                               "blocks.0.attn2.to_out.0",
                               "blocks.0.attn2.to_q",
                               "blocks.0.attn2.to_v",
                               "blocks.0.ffn.net.0.proj",
                               "blocks.0.ffn.net.2",
                               "blocks.0.norm2",
                               "blocks.0.scale_shift_table",
                               "blocks.38.scale_shift_table",
                               "blocks.39.attn1.norm_k",
                               "blocks.39.attn1.norm_q",
                               "blocks.39.attn1.to_k",
                               "blocks.39.attn1.to_out.0",
                               "blocks.39.attn1.to_q",
                               "blocks.39.attn1.to_v",
                               "blocks.39.attn2.norm_k",
                               "blocks.39.attn2.norm_q",
                               "blocks.39.attn2.to_k",
                               "blocks.39.attn2.to_out.0",
                               "blocks.39.attn2.to_q",
                               "blocks.39.attn2.to_v",
                               "blocks.39.ffn.net.0.proj",
                               "blocks.39.ffn.net.2",
                               "blocks.39.norm2",
                               "blocks.39.scale_shift_table"],
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(model_name,
                                            subfolder="transformer",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=transformer_quantization_config
                                            )
    generator = WanImageToVideoPipeline.from_pretrained(model_name,
                                                       vae=vae,
                                                       transformer=transformer,
                                                       text_encoder=text_encoder,
                                                       torch_dtype=torch.bfloat16
                                                       )
    generator.save_pretrained("../models/Wan2.1-T2V-14B")

def quantize_wan21_t2v_1_3b():
    model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    generator = WanImageToVideoPipeline.from_pretrained(model_name,
                                                       vae=vae,
                                                       text_encoder=text_encoder,
                                                       torch_dtype=torch.bfloat16
                                                       )
    generator.save_pretrained("../models/Wan2.1-T2V-1.3B")

def quantize_wan21_i2v_14b_480():
    model_name="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["time_embedder",
                               "timesteps_proj",
                               "time_proj",
                               "norm_out",
                               "proj_out",
                               "blocks.0.attn1.norm_k",
                               "blocks.0.attn1.norm_q",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_out.0",
                               "blocks.0.attn1.to_q",
                               "blocks.0.attn1.to_v",
                               "blocks.0.attn2.add_k_proj",
                               "blocks.0.attn2.add_v_proj",
                               "blocks.0.attn2.norm_added_q",
                               "blocks.0.attn2.norm_k",
                               "blocks.0.attn2.norm_q",
                               "blocks.0.attn2.to_k",
                               "blocks.0.attn2.to_out.0",
                               "blocks.0.attn2.to_q",
                               "blocks.0.attn2.to_v",
                               "blocks.0.ffn.net.0.proj",
                               "blocks.0.ffn.net.2",
                               "blocks.0.norm2",
                               "blocks.0.scale_shift_table",
                               "blocks.39.attn1.norm_k",
                               "blocks.39.attn1.norm_q",
                               "blocks.39.attn1.to_k",
                               "blocks.39.attn1.to_out.0",
                               "blocks.39.attn1.to_q",
                               "blocks.39.attn1.to_v",
                               "blocks.39.attn2.add_k_proj",
                               "blocks.39.attn2.add_v_proj",
                               "blocks.39.attn2.norm_added_k",
                               "blocks.39.attn2.norm_added_q",
                               "blocks.39.attn2.norm_k",
                               "blocks.39.attn2.norm_q",
                               "blocks.39.attn2.to_k",
                               "blocks.39.attn2.to_out.0",
                               "blocks.39.attn2.to_q",
                               "blocks.39.attn2.to_v",
                               "blocks.39.ffn.net.0.proj",
                               "blocks.39.ffn.net.2",
                               "blocks.39.norm2",
                               "blocks.39.scale_shift_table"
                               ],
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(model_name,
                                            subfolder="transformer",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=transformer_quantization_config
                                            )
    generator = WanImageToVideoPipeline.from_pretrained(model_name,
                                                       vae=vae,
                                                       transformer=transformer,
                                                       text_encoder=text_encoder,
                                                       torch_dtype=torch.bfloat16
                                                       )
    generator.save_pretrained("../models/Wan2.1-I2V-14B-480P")

def quantize_wan22_ti2v_5b():
    model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["time_embedder",
                               "timesteps_proj",
                               "time_proj",
                               "norm_out",
                               "proj_out",
                               "blocks.0.attn1.norm_k",
                               "blocks.0.attn1.norm_q",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_out.0",
                               "blocks.0.attn1.to_q",
                               "blocks.0.attn1.to_v",
                               "blocks.0.attn2.norm_k",
                               "blocks.0.attn2.norm_q",
                               "blocks.0.attn2.to_k",
                               "blocks.0.attn2.to_out.0",
                               "blocks.0.attn2.to_q",
                               "blocks.0.attn2.to_v",
                               "blocks.0.ffn.net.0.proj",
                               "blocks.0.ffn.net.2",
                               "blocks.0.norm2",
                               "blocks.0.scale_shift_table",
                               "blocks.29.attn1.norm_k",
                               "blocks.29.attn1.norm_q",
                               "blocks.29.attn1.to_k",
                               "blocks.29.attn1.to_out.0",
                               "blocks.29.attn1.to_q",
                               "blocks.29.attn1.to_v",
                               "blocks.29.attn2.norm_k",
                               "blocks.29.attn2.norm_q",
                               "blocks.29.attn2.to_k",
                               "blocks.29.attn2.to_out.0",
                               "blocks.29.attn2.to_q",
                               "blocks.29.attn2.to_v",
                               "blocks.29.ffn.net.0.proj",
                               "blocks.29.ffn.net.2",
                               "blocks.29.norm2",
                               "blocks.29.scale_shift_table"],
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanVACETransformer3DModel.from_pretrained(model_name,
                                                            subfolder="transformer",
                                                            torch_dtype=torch.bfloat16,
                                                            quantization_config=transformer_quantization_config
                                                            )
    generator = WanImageToVideoPipeline.from_pretrained(model_name,
                                                       vae=vae,
                                                       transformer=transformer,
                                                       text_encoder=text_encoder,
                                                       torch_dtype=torch.bfloat16
                                                       )
    generator.save_pretrained("../models/Wan2.2-TI2V-5B")

def quantize_wan21_vace_1_3b():
    model_name="Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["time_embedder",
                               "timesteps_proj",
                               "time_proj",
                               "norm_out",
                               "proj_out",
                               "blocks.0.attn1.norm_k",
                               "blocks.0.attn1.norm_q",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_out.0",
                               "blocks.0.attn1.to_q",
                               "blocks.0.attn1.to_v",
                               "blocks.0.attn2.norm_k",
                               "blocks.0.attn2.norm_q",
                               "blocks.0.attn2.to_k",
                               "blocks.0.attn2.to_out.0",
                               "blocks.0.attn2.to_q",
                               "blocks.0.attn2.to_v",
                               "blocks.0.ffn.net.0.proj",
                               "blocks.0.ffn.net.2",
                               "blocks.0.norm2",
                               "blocks.0.scale_shift_table",
                               "blocks.29.attn1.norm_k",
                               "blocks.29.attn1.norm_q",
                               "blocks.29.attn1.to_k",
                               "blocks.29.attn1.to_out.0",
                               "blocks.29.attn1.to_q",
                               "blocks.29.attn1.to_v",
                               "blocks.29.attn2.norm_k",
                               "blocks.29.attn2.norm_q",
                               "blocks.29.attn2.to_k",
                               "blocks.29.attn2.to_out.0",
                               "blocks.29.attn2.to_q",
                               "blocks.29.attn2.to_v",
                               "blocks.29.ffn.net.0.proj",
                               "blocks.29.ffn.net.2",
                               "blocks.29.norm2",
                               "blocks.29.scale_shift_table"
                               ],
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    )

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(model_name,
                                            subfolder="transformer",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=transformer_quantization_config
                                            )
    generator = WanVACEPipeline.from_pretrained(model_name,
                                                vae=vae,
                                                transformer=transformer,
                                                text_encoder=text_encoder,
                                                torch_dtype=torch.bfloat16
                                                )
    generator.save_pretrained("../models/Wan2.1-VACE-1.3B")

def quantize_wan22_i2v_a14b():
    model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    transformer_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["time_embedder",
                               "timesteps_proj",
                               "time_proj",
                               "norm_out",
                               "proj_out",
                               "blocks.0.attn1.norm_k",
                               "blocks.0.attn1.norm_q",
                               "blocks.0.attn1.to_k",
                               "blocks.0.attn1.to_out.0",
                               "blocks.0.attn1.to_q",
                               "blocks.0.attn1.to_v",
                               "blocks.0.attn2.norm_k",
                               "blocks.0.attn2.norm_q",
                               "blocks.0.attn2.to_k",
                               "blocks.0.attn2.to_out.0",
                               "blocks.0.attn2.to_q",
                               "blocks.0.attn2.to_v",
                               "blocks.0.ffn.net.0.proj",
                               "blocks.0.ffn.net.2",
                               "blocks.0.norm2",
                               "blocks.0.scale_shift_table",
                               "blocks.39.attn1.norm_k",
                               "blocks.39.attn1.norm_q",
                               "blocks.39.attn1.to_k",
                               "blocks.39.attn1.to_out.0",
                               "blocks.39.attn1.to_q",
                               "blocks.39.attn1.to_v",
                               "blocks.39.attn2.norm_k",
                               "blocks.39.attn2.norm_q",
                               "blocks.39.attn2.to_k",
                               "blocks.39.attn2.to_out.0",
                               "blocks.39.attn2.to_q",
                               "blocks.39.attn2.to_v",
                               "blocks.39.ffn.net.0.proj",
                               "blocks.39.ffn.net.2",
                               "blocks.39.norm2",
                               "blocks.39.scale_shift_table"]
    )
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(model_name,
                                                    subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=text_encoder_quantization_config
                                                    ).to("cpu")

    vae = AutoModel.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(model_name,
                                            subfolder="transformer",
                                            torch_dtype=torch.bfloat16,
                                            quantization_config=transformer_quantization_config
                                            ).to("cpu")
    transformer_2 = AutoModel.from_pretrained(model_name,
                                              subfolder="transformer_2",
                                              torch_dtype=torch.bfloat16,
                                              quantization_config=transformer_quantization_config
                                              )
    generator = WanImageToVideoPipeline.from_pretrained(model_name,
                                                        vae=vae,
                                                        transformer=transformer,
                                                        transformer_2=transformer_2,
                                                        text_encoder=text_encoder,
                                                        torch_dtype=torch.bfloat16
                                                        )
    generator.save_pretrained("../models/Wan2.2-I2V-A14B")

def quantize_llm(model_name=None):
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    if model_name is None:
        model_name = "Goekdeniz-Guelmez/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4"
    generator = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config)
    generator.save_pretrained("../models/Josiefied-Qwen2.5-14B-Instruct-abliterated-v4")

quantize_wan21_t2v_14b()