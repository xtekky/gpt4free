# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Ensure Python outputs everything immediately (useful for real-time logging in Docker).
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container.
WORKDIR /app

# Update the system packages and install system-level dependencies required for compilation.
# gcc: Compiler required for some Python packages.
# build-essential: Contains necessary tools and libraries for building software.
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy the project's requirements file into the container.
COPY requirements.txt /app/

# Upgrade pip for the latest features and install the project's Python dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into the container.
# This may include all code, assets, and configuration files required to run the application.
COPY . /app/

# Install additional requirements specific to the interference module/package.
RUN pip install -r etc/interference/requirements.txt

# Expose port 1337
EXPOSE 1337

# Define the default command to run the app using Python's module mode.
CMD ["python", "-m", "etc.interference.app"]