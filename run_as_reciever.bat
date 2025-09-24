@echo off
echo [INFO] Starting the U-LANC client as the RECEIVER (Answerer)...
call "venv\Scripts\activate.bat"
python client.py receive
pause