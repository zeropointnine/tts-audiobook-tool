@echo off
echo Activating virtual environment...
call venv-chatterbox\Scripts\activate && python -m tts_audiobook_tool

echo System is ready!
:: This keeps the window open and the venv active
cmd /k