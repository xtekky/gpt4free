@echo off

echo Setting G4F_PROXY environment variable...
setx G4F_PROXY "http://127.0.0.1:7890"

echo Running the application...
cd gpt4free
conda activate gpt4free && python g4f/gui/run.py
