# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

It is fairly basic with only a few endpoints at moment which can be documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

# Install:
Run the install script for either windows or linux depending on your operating system.

Start via `start_avernus.sh` or `start_avernus.bat` dependiing on your operating system. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

# TODO:

- Add WAN2.1 support (waiting on diffusers update)
- Add retrieval augmented generation support.
- Add audio generation endpoints
- Add svg model endpoint
- Return different status codes if a generation ooms

