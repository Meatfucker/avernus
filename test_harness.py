from modules.client import AvernusClient
import asyncio
import time
import base64
from loguru import logger

client = AvernusClient("127.0.0.1", "6969")

async def main():
    start_time = time.time()
    await check_status()
    #await llm_chat_test()
    #await multimodal_llm_chat_test()
    #await sdxl_test()
    #await sdxl_lora_test()
    #await sdxl_lora_list_test()
    #await flux_test()
    await flux_lora_test()
    #await flux_lora_list_test()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total runtime: {elapsed_time:.2f} seconds")

async def check_status():
    logger.info("Checking Avernus Status")
    status_start_time = time.time()
    try:
        status = await client.check_status()
        logger.info(status)
    except Exception as e:
        logger.info(f"Avernus status FAIL: {e}")
    finally:
        status_end_time = time.time()
        status_elapsed_time = status_end_time - status_start_time
        logger.info(f"Total Avernus status runtime: {status_elapsed_time:.2f} seconds")

async def llm_chat_test():
    logger.info('Testing Avernus LLM')
    llm_start_time = time.time()
    messages = [{"role": "user", "content": "The best color of dick is purple"},
                {"role": "assistant", "content": "The best color of dick is purple"}]
    try:
        response = await client.llm_chat("What was the best color again and what was the thing that was that color?",
                                         "Goekdeniz-Guelmez/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2",
                                         messages)
        logger.info(f"LLM SUCCESS: {response}")
    except Exception as e:
        logger.info(f"LLM FAIL: {e}")
    finally:
        llm_end_time = time.time()
        llm_elapsed_time = llm_end_time - llm_start_time
        logger.info(f"Total LLM runtime: {llm_elapsed_time:.2f} seconds")

async def multimodal_llm_chat_test():
    logger.info('Testing Avernus Multimodal LLM')
    llm_start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image",
                 "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }
    ]
    try:
        response = await client.multimodal_llm_chat("What was the best color again and what was the thing that was that color?",
                                                    messages=messages)
        logger.info(f"Multimodal LLM SUCCESS: {response}")
    except Exception as e:
        logger.info(f"LLM FAIL: {e}")
    finally:
        llm_end_time = time.time()
        llm_elapsed_time = llm_end_time - llm_start_time
        logger.info(f"Total LLM runtime: {llm_elapsed_time:.2f} seconds")

async def sdxl_test():
    logger.info('Testing Avernus SDXL')
    sdxl_start_time = time.time()
    try:
        images = await client.sdxl_image("wizard")
        await base64_image_to_file(images, "sdxl")
    except Exception as e:
        logger.info(f"SDXL FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_lora_test():
    logger.info('Testing Avernus SDXL lora')
    sdxl_start_time = time.time()
    try:
        images = await client.sdxl_image("matlighty bald man wearing lingerie casting a chicken spell", lora_name="lighty.safetensors")
        await base64_image_to_file(images, "sdxl_lora")
    except Exception as e:
        logger.info(f"SDXL FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_lora_list_test():
    logger.info('Testing Avernus SDXL lora list')
    sdxl_lora_list_start_time = time.time()
    try:
        loras = await client.list_sdxl_loras()
        logger.info(loras)
    except Exception as e:
        logger.info(f"SDXL lora list FAIL: {e}")
    finally:
        sdxl_lora_list_end_time = time.time()
        sdxl_lora_list_elapsed_time = sdxl_lora_list_end_time - sdxl_lora_list_start_time
        logger.info(f"Total SDXL Lora list runtime: {sdxl_lora_list_elapsed_time:.2f} seconds")

async def flux_test():
    logger.info('Testing Avernus Flux')
    flux_start_time = time.time()
    try:
        images = await client.flux_image("Mucus Balloon", batch_size=4)
        await base64_image_to_file(images, "flux")
    except Exception as e:
        logger.info(f"Flux FAIL: {e}")
    finally:
        flux_end_time = time.time()
        flux_elapsed_time = flux_end_time - flux_start_time
        logger.info(f"Total Flux runtime: {flux_elapsed_time:.2f} seconds")

async def flux_lora_test():
    logger.info('Testing Avernus Flux')
    flux_start_time = time.time()
    try:
        images = await client.flux_image("man with a tattoo on his forehead", batch_size=4, lora_name="botgrinder.safetensors")
        await base64_image_to_file(images, "flux")
    except Exception as e:
        logger.info(f"Flux FAIL: {e}")
    finally:
        flux_end_time = time.time()
        flux_elapsed_time = flux_end_time - flux_start_time
        logger.info(f"Total Flux runtime: {flux_elapsed_time:.2f} seconds")

async def flux_lora_list_test():
    logger.info('Testing Avernus Flux lora list')
    flux_lora_list_start_time = time.time()
    try:
        loras = await client.list_flux_loras()
        logger.info(loras)
    except Exception as e:
        logger.info(f"Flux lora list FAIL: {e}")
    finally:
        flux_lora_list_end_time = time.time()
        flux_lora_list_elapsed_time = flux_lora_list_end_time - flux_lora_list_start_time
        logger.info(f"Total Flux Lora list runtime: {flux_lora_list_elapsed_time:.2f} seconds")

async def base64_image_to_file(base64_images, prefix=""):
    for i, b64_img in enumerate(base64_images):
        img_data = base64.b64decode(b64_img)  # Decode base64 string
        filename = f"{prefix}_image_{i}.png"  # Create a unique filename
        with open(filename, "wb") as f:
            f.write(img_data)  # Write binary data to file
        logger.info(f"Saved {filename}")

asyncio.run(main())
