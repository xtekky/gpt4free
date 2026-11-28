FROM selenium/node-chrome

ARG G4F_VERSION
ARG G4F_GO_VERSION
ENV G4F_VERSION $G4F_VERSION

ENV SE_SCREEN_WIDTH 1850
ENV G4F_DIR /app
# API server port; the Classic desktop/noVNC login stays on 7900 (G4F_LOGIN_URL),
# so keep these two on different container ports.
ENV G4F_API_PORT 8080
ENV G4F_LOGIN_URL http://localhost:7900/?autoconnect=1&resize=scale&password=secret
# g4f-go provisions its embedded CPython runtime here; world-readable so the
# non-root server user can run python/pip/python3.
ENV G4F_CONFIG_DIR=/opt/g4f
ENV PATH=${G4F_CONFIG_DIR}/python-embed/python-home/bin:$PATH

USER root

#  If docker compose, install git
RUN if [ "$G4F_VERSION" = "" ] ; then \
  apt-get -qqy update && \
  apt-get -qqy install git \
  ; fi

# Install Python3, pip, remove OpenJDK 11, clean up
RUN apt-get -qqy update \
  && apt-get -qqy upgrade \
  && apt-get -qyy autoremove \
  && apt-get -qqy install curl unzip python3 python-is-python3 pip ffmpeg flac libavcodec-extra \
  && apt-get -qyy remove openjdk-11-jre-headless \
  && apt-get -qyy autoremove \
  && apt-get -qyy clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Update entrypoint
COPY docker/supervisor.conf /etc/supervisor/conf.d/selenium.conf
COPY docker/supervisor-api.conf /etc/supervisor/conf.d/api.conf

# Change background image
COPY docker/background.png /usr/share/images/fluxbox/ubuntu-light.png

# Add user, fix permissions
RUN chown "${SEL_UID}:${SEL_GID}" $HOME/.local

# Switch user
USER $SEL_UID

# Set the working directory in the container.
WORKDIR $G4F_DIR

# Copy the project source into the container (as root so g4f-go can provision
# its shared python runtime into /opt/g4f below).
COPY setup.py README.md MANIFEST.in $G4F_DIR/
COPY requirements.txt $G4F_DIR
ADD g4f $G4F_DIR/g4f

# Install g4f-go and provision its embedded CPython runtime as the container's
# python/pip/python3. g4f is installed into g4f-go's runtime via an editable
# install of the local checkout, and the project's requirements are installed
# on top, so `python -m g4f api` runs on g4f-go python using the local g4f.
USER root
RUN set -eux; \
    go_arch="$(dpkg --print-architecture)"; \
    case "$go_arch" in amd64|arm64) ;; *) echo "unsupported arch $go_arch" >&2; exit 1 ;; esac; \
    version="${G4F_GO_VERSION:-}"; \
    if [ -z "$version" ]; then \
      version="$(curl -fsSL --retry 2 "https://api.github.com/repos/xtekky/gpt4free/releases/latest" | grep -m1 '"tag_name":' | sed -E 's/.*"tag_name": *"?([^,"]+)".*/\1/')"; \
    fi; \
    case "$version" in v*) ;; *) version="v$version" ;; esac; \
    [ -n "$version" ] || { echo "could not resolve g4f-go version (set G4F_GO_VERSION)" >&2; exit 1; }; \
    curl -fsSL --retry 3 -o /tmp/g4f-go.zip "https://github.com/xtekky/gpt4free/releases/download/${version}/g4f-go-${version}-linux-${go_arch}.zip"; \
    unzip -qo /tmp/g4f-go.zip -d /tmp/g4f-go; \
    install -m 0755 "$(find /tmp/g4f-go -type f -name g4f-go | head -1)" /usr/local/bin/g4f-go; \
    rm -rf /tmp/g4f-go /tmp/g4f-go.zip; \
    g4f-go install -e /app; \
    "${G4F_CONFIG_DIR}/python-embed/python-home/bin/pip" install --no-cache-dir -r /app/requirements.txt; \
    ln -sf "${G4F_CONFIG_DIR}/python-embed/python-home/bin/python3.15" /usr/local/bin/python3; \
    ln -sf "${G4F_CONFIG_DIR}/python-embed/python-home/bin/python3.15" /usr/local/bin/python; \
    ln -sf "${G4F_CONFIG_DIR}/python-embed/python-home/bin/pip3" /usr/local/bin/pip3; \
    ln -sf "${G4F_CONFIG_DIR}/python-embed/python-home/bin/pip3" /usr/local/bin/pip; \
    chmod -R a+rX "${G4F_CONFIG_DIR}"; \
    g4f-go status || true

# Switch back to the unprivileged server user and fix source ownership.
USER $SEL_UID
RUN chown -R "${SEL_UID}:${SEL_GID}" $G4F_DIR

# Copy the Android build commands into the container so the Android toolchain
# (JDK 17, Gradle 8.7, Android SDK 34, NDK, CMake) can be installed on demand
# and the APK can be built on demand:
#   docker compose run --rm gpt4free /app/bin/install-android.sh   # install toolchain only
#   docker compose run --rm gpt4free /app/bin/build-android.sh    # build the APK
COPY bin/build-android.sh bin/install-android.sh $G4F_DIR/bin/
RUN chmod +x $G4F_DIR/bin/build-android.sh \
        $G4F_DIR/bin/install-android.sh

# Expose ports
EXPOSE 8080 7900