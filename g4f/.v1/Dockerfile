FROM python:3.11.3-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

COPY . /root/gpt4free

WORKDIR /root/gpt4free

CMD ["streamlit", "run", "./gui/streamlit_app.py"]

EXPOSE 8501
