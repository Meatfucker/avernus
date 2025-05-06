@echo off
set VENV_DIR=venv
call %VENV_DIR%\Scripts\activate.bat
python modules/rag.py
pause
