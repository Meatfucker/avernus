# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

It is fairly basic with only a few endpoints at moment which can be documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

# Install:
Create a python environment using either python venv or conda.

Enter the environment

Install the requirements file using pip.

Start via `start_avernus.sh` after entering the venv or conda environment. I recommend against installing python packages for this program outside of an environment. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.

# TODO:

- Add proper installer and startup script for windows and linux
- Add an endpoint to list local models.
- Add an endpoint to list local loras.
- Add WAN2.1 support (waiting on diffusers update)
- Add retrieval augmented generation support.
- Add audio generation endpoints
- Add svg model endpoint

