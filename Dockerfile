FROM python:3.11 as builder

WORKDIR /usr/app
ENV PATH="/usr/app/venv/bin:$PATH"

#RUN apt-get update && apt-get install -y git
RUN apt-get update
RUN apt-get install ffmpeg #issue 445

RUN mkdir -p /usr/app
RUN python -m venv ./venv

COPY requirements.txt .

RUN pip install -r requirements.txt

# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# RUN pip config set global.trusted-host mirrors.aliyun.com

FROM python:3.11

WORKDIR /usr/app
ENV PATH="/usr/app/venv/bin:$PATH"

COPY --from=builder /usr/app/venv ./venv
COPY . .

RUN cp ./gui/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]

EXPOSE 8501
