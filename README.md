# avernus

This is a ml inference server based on huggingface libraries which provides a REST based API for client interactions.

It is fairly basic with only a few endpoints at moment which can be documented by accessing the server at http://127.0.0.1:6969/docs if using the default port.

requirements file is not well tested across anything other than Ubuntu and Mint. Should work on other distros but some manual pip package installs may be required.

Start via `start_avernus.sh` after entering the venv or conda environment. I recommend against installing python packages for this program outside of an environment. The startup script also contains the url and port of the avernus server, if you want to change the port or bind to a specific device.