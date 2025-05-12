# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

It is fairly basic with only a few endpoints at moment which can be documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

There is a basic python client available in modules/client.py

# Install:
Run the install script for either windows or linux depending on your operating system.

Start via `start_avernus.sh` or `start_avernus.bat` dependiing on your operating system. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

If you want to use the rag endpoint you need to place urls, one per line in rag/urls. Document files go in the rag/docs directory and then run create_embeddings.sh to create the embeddings. This only needs to done once each time you change the urls.

# Endpoints:

- /status - Returns server status
- /llm_chat - For generating llm responses
- /sdxl_generate - For generating sdxl images
- /flux_generate - For generating flux images
- /list_sdxl_loras - Returns a list of available loras from loras/sdxl
- /list_sdxl_controlnets - Returns a list of available SDXL controlnets
- /list_flux_loras - Returns a list of available loras from loras/flux
- /rag_retrieve - Returns relevant rag documents to the query

# TODO:

- Add WAN2.1 support (waiting on me to figure out memory issues)
- Add audio generation endpoints
- Add svg model endpoint
- Return different status codes if a generation ooms

