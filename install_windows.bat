@echo off
:: Name of the virtual environment folder
set VENV_DIR=venv

:: Create a virtual environment
python -m venv %VENV_DIR%

:: Activate the virtual environment
call %VENV_DIR%\Scripts\activate.bat

:: Install the requirements
pip install -r requirements-windows.txt

:: Keep the command prompt open if you want to see the output
pause
