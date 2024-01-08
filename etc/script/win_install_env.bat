@echo off

echo Installing Miniconda...
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /SILENT /AddToPath=1
del miniconda.exe

echo Installing Git for Windows...
curl -L https://mirror.ghproxy.com/https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe -o git-install.exe
start /wait "" git-install.exe /SILENT /AddToPath=1
del git-install.exe

echo Cloning the repository...
git clone https://mirror.ghproxy.com/https://github.com/xtekky/gpt4free
cd gpt4free
conda  activate gpt4free && pip install -r requirements.txt && pip install g4f
