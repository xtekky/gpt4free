FROM selenium/node-chrome

ENV SE_SCREEN_WIDTH 1850
ENV G4F_LOGIN_URL http://localhost:7900/?autoconnect=1&resize=scale&password=secret
ENV PYTHONUNBUFFERED 1
ARG G4F_VERSION
ENV G4F_VERSION ${G4F_VERSION}

USER root

# Python packages
RUN apt-get -qqy update \
  && apt-get -qqy install \
    python3 \
    python-is-python3 \
    pip

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
  && apt-get -qyy autoremove \
  && apt-get -qyy clean

# Update entrypoint
COPY docker/supervisor.conf /etc/supervisor/conf.d/selenium.conf

# Change background image
COPY docker/background.png /usr/share/images/fluxbox/ubuntu-light.png

# Switch user
USER 1200

# Set the working directory in the container.
WORKDIR /app

# Copy the project's requirements file into the container.
COPY requirements.txt /app/

# Upgrade pip for the latest features and install the project's Python dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire package into the container.
COPY g4f /app/g4f

# Expose ports
EXPOSE 80 1337