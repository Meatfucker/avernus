#!/bin/bash
VENV_DIR="venv"
source $VENV_DIR/bin/activate
uvicorn avernus:avernus --host 0.0.0.0 --port 6969
