#!/bin/bash

# Start the pulseaudio server
pulseaudio -D --exit-idle-time=-1

# Load the virtual sink and set it as default
pacmd load-module module-virtual-sink sink_name=v1
pacmd set-default-sink v1

# Set the monitor of v1 sink to be the default source
pacmd set-default-source v1.monitor

rm -f /tmp/.X*lock

# Start app servers
python -m g4f.cli api &
python -m g4f.cli gui