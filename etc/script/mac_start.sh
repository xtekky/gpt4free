#! /bin/bash
echo Running the application...
cd gpt4free
export G4F_PROXY=http://127.0.0.1:7890
source activate gpt4free && python g4f/gui/run.py
