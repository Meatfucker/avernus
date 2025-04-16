from modules.client import AvernusClient
import asyncio
import time
import base64
from io import BytesIO
from loguru import logger
from PIL import Image

client = AvernusClient("127.0.0.1", "6969")

async def main():
    start_time = time.time()
    await check_status()
    #await llm_chat_test()
    #await multimodal_llm_chat_test()
    #await sdxl_test()
    #await sdxl_lora_test()
    #await sdxl_i2i_test()
    #await sdxl_i2i_lora_test()
    #await sdxl_lora_list_test()
    #await flux_test()
    #await flux_lora_test()
    #await flux_i2i_test()
    #await flux_lora_i2i_test()
    #await flux_lora_list_test()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total runtime: {elapsed_time:.2f} seconds")

async def check_status():
    logger.info("Checking Avernus Status")
    status_start_time = time.time()
    try:
        status = await client.check_status()
        logger.success(status)
    except Exception as e:
        logger.error(f"Avernus status FAIL: {e}")
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
        logger.success(f"LLM SUCCESS: {response}")
    except Exception as e:
        logger.error(f"LLM FAIL: {e}")
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
        logger.success(f"Multimodal LLM SUCCESS: {response}")
    except Exception as e:
        logger.error(f"LLM FAIL: {e}")
    finally:
        llm_end_time = time.time()
        llm_elapsed_time = llm_end_time - llm_start_time
        logger.info(f"Total LLM runtime: {llm_elapsed_time:.2f} seconds")

async def sdxl_test():
    logger.info('Testing Avernus SDXL')
    sdxl_start_time = time.time()
    try:
        images = await client.sdxl_image("wizard",
                                         batch_size=2)
        await base64_image_to_file(images, "sdxl")
        logger.success("SDXL SUCCESS")
    except Exception as e:
        logger.error(f"SDXL FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_i2i_test():
    logger.info('Testing Avernus SDXL I2I')
    sdxl_start_time = time.time()
    try:
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await client.sdxl_image("warrior princess",
                                         image=image,
                                         strength=0.7,
                                         batch_size=2)
        await base64_image_to_file(images, "sdxl_i2i")
        logger.success("SDXL I2I SUCCESS")
    except Exception as e:
        logger.info(f"SDXL I2I FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL I2I runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_i2i_lora_test():
    logger.info('Testing Avernus SDXL LORA I2I')
    sdxl_start_time = time.time()
    try:
        image = Image.open("tests/sdxl_image_0.png")
        image = image_to_base64(image)
        images = await client.sdxl_image("matlighty man robot hooker",
                                         image=image,
                                         strength=0.7,
                                         lora_name="lighty.safetensors",
                                         batch_size=2)
        await base64_image_to_file(images, "sdxl_i2i_lora")
        logger.success("SDXL LORA I2I SUCCESS")
    except Exception as e:
        logger.error(f"SDXL LORA I2I FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL LORA I2I runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_lora_test():
    logger.info('Testing Avernus SDXL lora')
    sdxl_start_time = time.time()
    try:
        images = await client.sdxl_image("matlighty bald man wearing lingerie casting a chicken spell",
                                         lora_name="lighty.safetensors",
                                         batch_size=2)
        await base64_image_to_file(images, "sdxl_lora")
        logger.success("SDXL LORA SUCCESS")
    except Exception as e:
        logger.error(f"SDXL LORA FAIL: {e}")
    finally:
        sdxl_end_time = time.time()
        sdxl_elapsed_time = sdxl_end_time - sdxl_start_time
        logger.info(f"Total SDXL LORA runtime: {sdxl_elapsed_time:.2f} seconds")

async def sdxl_lora_list_test():
    logger.info('Testing Avernus SDXL lora list')
    sdxl_lora_list_start_time = time.time()
    try:
        loras = await client.list_sdxl_loras()
        logger.success(loras)
    except Exception as e:
        logger.error(f"SDXL LORA LIST FAIL: {e}")
    finally:
        sdxl_lora_list_end_time = time.time()
        sdxl_lora_list_elapsed_time = sdxl_lora_list_end_time - sdxl_lora_list_start_time
        logger.info(f"Total SDXL Lora list runtime: {sdxl_lora_list_elapsed_time:.2f} seconds")

async def flux_test():
    logger.info('Testing Avernus Flux')
    flux_start_time = time.time()
    try:
        images = await client.flux_image("Mucus Balloon",
                                         batch_size=2)
        await base64_image_to_file(images, "flux")
        logger.success("FLUX SUCCESS")
    except Exception as e:
        logger.error(f"FLUX FAIL: {e}")
    finally:
        flux_end_time = time.time()
        flux_elapsed_time = flux_end_time - flux_start_time
        logger.info(f"Total Flux runtime: {flux_elapsed_time:.2f} seconds")

async def flux_lora_test():
    logger.info('Testing Avernus Flux Lora')
    flux_start_time = time.time()
    try:
        images = await client.flux_image("man with a tattoo on his forehead",
                                         batch_size=2,
                                         lora_name="lighty_peft.safetensors")
        await base64_image_to_file(images, "flux_lora")
        logger.success("FLUX LORA SUCCESS")
    except Exception as e:
        logger.error(f"FLUX LORA FAIL: {e}")
    finally:
        flux_end_time = time.time()
        flux_elapsed_time = flux_end_time - flux_start_time
        logger.info(f"Total Flux Lora runtime: {flux_elapsed_time:.2f} seconds")

async def flux_i2i_test():
    logger.info('Testing Avernus Flux I2I')
    flux_i2i_start_time = time.time()
    try:
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        images = await client.flux_image("basketball",
                                         image=image,
                                         batch_size=2,
                                         strength=0.7)
        await base64_image_to_file(images, "flux_i2i")
        logger.success("FLUX I2I SUCCESS")
    except Exception as e:
        logger.error(f"FLUX I2I FAIL: {e}")
    finally:
        flux_i2i_end_time = time.time()
        flux_i2i_elapsed_time = flux_i2i_end_time - flux_i2i_start_time
        logger.info(f"Total Flux i2i runtime: {flux_i2i_elapsed_time:.2f} seconds")

async def flux_lora_i2i_test():
    logger.info('Testing Avernus Flux Lora I2I')
    flux_i2i_start_time = time.time()
    try:
        image = Image.open("tests/flux_image_0.png")
        image = image_to_base64(image)
        images = await client.flux_image("matlighty man monstrous creature",
                                         image=image,
                                         batch_size=2,
                                         strength=0.9,
                                         lora_name="lighty_peft.safetensors")
        await base64_image_to_file(images, "flux_lora_i2i")
        logger.success("FLUX LORA I2I SUCCESS")
    except Exception as e:
        logger.error(f"FLUX LORA I2I FAIL: {e}")
    finally:
        flux_i2i_end_time = time.time()
        flux_i2i_elapsed_time = flux_i2i_end_time - flux_i2i_start_time
        logger.info(f"Total Flux Lora i2i runtime: {flux_i2i_elapsed_time:.2f} seconds")

async def flux_lora_list_test():
    logger.info('Testing Avernus Flux lora list')
    flux_lora_list_start_time = time.time()
    try:
        loras = await client.list_flux_loras()
        logger.success(loras)
    except Exception as e:
        logger.error(f"Flux lora list FAIL: {e}")
    finally:
        flux_lora_list_end_time = time.time()
        flux_lora_list_elapsed_time = flux_lora_list_end_time - flux_lora_list_start_time
        logger.info(f"Total Flux Lora list runtime: {flux_lora_list_elapsed_time:.2f} seconds")

async def base64_image_to_file(base64_images, prefix=""):
    for i, b64_img in enumerate(base64_images):
        img_data = base64.b64decode(b64_img)  # Decode base64 string
        filename = f"tests/{prefix}_image_{i}.png"  # Create a unique filename
        with open(filename, "wb") as f:
            f.write(img_data)  # Write binary data to file

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

asyncio.run(main())
