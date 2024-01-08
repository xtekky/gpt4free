#! /bin/bash

echo Installing Miniconda...
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh 
chmod +x Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
./Miniconda3-py38_22.11.1-1-Linux-x86_64.sh -b && /root/miniconda3/bin/conda init bash && /root/miniconda3/bin/conda clean -ya
echo Cloning the repository...
git clone https://mirror.ghproxy.com/https://github.com/xtekky/gpt4free
cd gpt4free
conda create -n gpt4free python=3.9
source activate gpt4free
pip install -r requirements.txt && pip install g4f
