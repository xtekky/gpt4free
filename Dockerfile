FROM python:3.11 as builder

WORKDIR /usr/app
ENV PATH="/usr/app/venv/bin:$PATH"
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git
RUN mkdir -p /usr/app
RUN python -m venv ./venv

COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com
RUN pip install -r requirements.txt
RUN pip install --upgrade numpy

# FROM python:3.11-alpinei

FROM python:3.11
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

RUN pip install numpy && pip install setuptools

WORKDIR /usr/app
ENV PATH="/usr/app/venv/bin:$PATH"

COPY --from=builder /usr/app/venv ./venv
COPY . .

RUN cp ./gui/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]

EXPOSE 8501
