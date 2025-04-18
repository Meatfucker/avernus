#!/bin/bash

# Name of the virtual environment folder
VENV_DIR="venv"

# Create a virtual environment
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install the requirements
uvicorn avernus:avernus --host 0.0.0.0 --port 6969
