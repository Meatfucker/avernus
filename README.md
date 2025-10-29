# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

Endpoints are documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

There is a basic python client available in modules/client.py

# Features:

- LLM chat support.
- SDXL support for multiple models, image2image, controlnets, and ipadapters.
- Flux Support. Only dev is currently supported. image2image and ipadapters are supported.
- Qwen Image, Qwen Image Edit, and Qwen Image Edit Plus
- Wan T2V and I2V
- ACE Step
- Chroma 
- HiDream
- Hunyuan Video
- Auraflow
- Lumina 2
- Upscalers
- SD 1.5
- Sana Sprint

# Install:
Run the install script for either windows or linux depending on your operating system.

Start via `start_avernus.sh` or `start_avernus.bat` depending on your operating system. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

# TODO:

- Add svg model endpoint
- Add LLM chat templates
- 3d model model support