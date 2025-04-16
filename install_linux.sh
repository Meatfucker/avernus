#!/bin/bash

# Name of the virtual environment folder
VENV_DIR="venv"

# Create a virtual environment
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install the requirements
pip install -r requirements.txt

# Keep the terminal open (Optional)
echo "Installation complete. Press Enter to exit."
read
