@echo off
set VENV_DIR=venv
call %VENV_DIR%\Scripts\activate.bat
uvicorn avernus:avernus --host 0.0.0.0 --port 6969
pause
