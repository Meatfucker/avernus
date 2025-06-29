# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

It is fairly basic with only a few endpoints at moment which can be documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

There is a basic python client available in modules/client.py

# Features:

- A fairly simple LLM chat endpoint. Supports model switching but not chat templates yet
- Basic rag support which is detailed in the install instructions.
- SDXL support for multiple models, image2image, controlnets, and ipadapters.
- Flux Support. Only dev is currently supported. image2image and controlnets are supported. ipadapters are implemented but waiting on an upstream bugfix to work.

# Install:
Run the install script for either windows or linux depending on your operating system.

Start via `start_avernus.sh` or `start_avernus.bat` dependiing on your operating system. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

If you want to use the rag endpoint you need to place urls, one per line in rag/urls. Document files go in the rag/docs directory and then run create_embeddings.sh to create the embeddings. This only needs to done once each time you change the urls.

# Endpoints:

- /status - Returns server status
- /llm_chat - For generating LLM responses
- /sdxl_generate - For generating SDXL images
- /sdxl_inpaint_generate - For SDXL inpainting
- /flux_generate - For generating Flux images
- /flux_inpaint_generate - For Flux inpainting
- /flux_kontext_generate - Flux Kontext
- /list_sdxl_loras - Returns a list of available loras from loras/sdxl
- /list_sdxl_controlnets - Returns a list of available SDXL controlnets
- /list_flux_loras - Returns a list of available loras from loras/flux
- /rag_retrieve - Returns relevant RAG documents to the query

# TODO:

- Add WAN2.1 support
- Add audio generation endpoints
- Add svg model endpoint
- Return different status codes if a generation ooms
- Add LLM chat templates