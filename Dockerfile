FROM python:3.10

RUN apt-get update && apt-get install -y git

RUN mkdir -p /usr/src/gpt4free
WORKDIR /usr/src/gpt4free

# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# RUN pip config set global.trusted-host mirrors.aliyun.com

COPY requirements.txt /usr/src/gpt4free/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/gpt4free
RUN cp gui/streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
