#!/bin/bash
while true; do
    sleep 5
    rm ~/.config/g4f/cookies/.nodriver_is_open
    google-chrome --remote-allow-origins=* --no-first-run --no-service-autorun --no-default-browser-check --homepage=about:blank --no-pings --password-store=basic --disable-infobars --disable-breakpad --disable-dev-shm-usage --disable-session-crashed-bubble --disable-search-engine-choice-screen --user-data-dir="~/.config/g4f-nodriver" --disable-session-crashed-bubble --disable-features=IsolateOrigins,site-per-process --remote-debugging-host=127.0.0.1 --remote-debugging-port=57011
done