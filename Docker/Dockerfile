FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/xtekky/gpt4free.git
WORKDIR /gpt4free
RUN pip install --no-cache-dir -r requirements.txt
RUN cp gui/streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
