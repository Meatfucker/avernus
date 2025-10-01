# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

Endpoints are documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

There is a basic python client available in modules/client.py

# Features:

- LLM chat support.
- SDXL support for multiple models, image2image, controlnets, and ipadapters.
- Flux Support. Only dev is currently supported. image2image and controlnets are supported. ipadapters are implemented but waiting on an upstream bugfix to work.
- Qwen Image and Qwen Image Edit support
- Wan T2V and I2V Support
- ACE Step Support

# Install:
Run the install script for either windows or linux depending on your operating system.

Start via `start_avernus.sh` or `start_avernus.bat` depending on your operating system. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

# TODO:

- add hidream support
- Add svg model endpoint
- Return different status codes if a generation ooms
- Add LLM chat templates
- Make api responsive while processing gen
- add model picker to flux
- LTX Support