FROM seleniarm/node-chromium

ARG G4F_VERSION
ARG G4F_USER=g4f
ARG G4F_USER_ID=1000
ARG G4F_NO_GUI
ARG G4F_PASS=secret

ENV G4F_VERSION $G4F_VERSION
ENV G4F_USER $G4F_USER
ENV G4F_USER_ID $G4F_USER_ID
ENV G4F_NO_GUI $G4F_NO_GUI

ENV SE_SCREEN_WIDTH 1850
ENV PYTHONUNBUFFERED 1
ENV G4F_DIR /app
ENV G4F_LOGIN_URL http://localhost:7900/?autoconnect=1&resize=scale&password=$G4F_PASS
ENV HOME /home/$G4F_USER
ENV PATH $PATH:$HOME/.local/bin
ENV SE_DOWNLOAD_DIR $HOME/Downloads
ENV SEL_USER $G4F_USER
ENV SEL_UID $G4F_USER_ID
ENV SEL_GID $G4F_USER_ID

USER root

#  If docker compose, install git
RUN if [ "$G4F_VERSION" = "" ] ; then \
  apt-get -qqy update && \
  apt-get -qqy install git \
  ; fi

# Install Python3, pip, remove OpenJDK 11, clean up
RUN apt-get -qqy update \
  && apt-get -qqy install python3 python-is-python3 pip \
  && apt-get -qyy remove openjdk-11-jre-headless \
  && apt-get -qyy autoremove \
  && apt-get -qyy clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Update entrypoint
COPY docker/supervisor.conf /etc/supervisor/conf.d/selenium.conf
COPY docker/supervisor-gui.conf /etc/supervisor/conf.d/gui.conf

# If no gui
RUN if [ "$G4F_NO_GUI" ] ; then \
  rm /etc/supervisor/conf.d/gui.conf \
  ; fi

# Change background image
COPY docker/background.png /usr/share/images/fluxbox/ubuntu-light.png

# Add user, fix permissions
RUN groupadd -g $G4F_USER_ID $G4F_USER \
  && useradd -rm -G sudo -u $G4F_USER_ID -g $G4F_USER_ID $G4F_USER \
  && echo "${G4F_USER}:${G4F_PASS}" | chpasswd \
  && mkdir "${SE_DOWNLOAD_DIR}" \
  && chown "${G4F_USER_ID}:${G4F_USER_ID}" $SE_DOWNLOAD_DIR /var/run/supervisor /var/log/supervisor \
  && chown "${G4F_USER_ID}:${G4F_USER_ID}" -R /opt/bin/ /usr/bin/chromedriver /opt/selenium/

# Switch user
USER $G4F_USER_ID

# Set VNC password
RUN mkdir -p ${HOME}/.vnc \
  && x11vnc -storepasswd ${G4F_PASS} ${HOME}/.vnc/passwd

# Set the working directory in the container.
WORKDIR $G4F_DIR

# Copy the project's requirements file into the container.
COPY requirements.txt $G4F_DIR

# Upgrade pip for the latest features and install the project's Python dependencies.
RUN pip install --break-system-packages --upgrade pip \
  && pip install --break-system-packages -r requirements.txt \
  && pip install --break-system-packages \
    undetected-chromedriver selenium-wire \
  && pip uninstall -y --break-system-packages \
    pywebview plyer

# Copy the entire package into the container.
ADD --chown=$G4F_USER:$G4F_USER g4f $G4F_DIR/g4f

# Expose ports
EXPOSE 8080 1337
