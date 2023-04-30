# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Install necessary packages and tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/xtekky/gpt4free.git /gpt4free

# Set the working directory
WORKDIR /gpt4free

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the Streamlit app file
RUN cp gui/streamlit_app.py .

# Start a new stage to reduce the final image size
FROM python:3.10-slim

# Install any needed packages specified in requirements.txt
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy the Streamlit app file from the previous stage
COPY --from=builder /gpt4free/streamlit_app.py /gpt4free/streamlit_app.py

# Expose the Streamlit port
EXPOSE 8501

# Run the command to start the Streamlit app
CMD ["streamlit", "run", "/gpt4free/streamlit_app.py"]
