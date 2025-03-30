@echo off
:: Name of the virtual environment folder
set VENV_DIR=venv

:: Activate the virtual environment
call %VENV_DIR%\Scripts\activate.bat

:: Run the Uvicorn server
uvicorn avernus:avernus --host 0.0.0.0 --port 6969

:: Keep the command prompt open if you want to see the output
pause
