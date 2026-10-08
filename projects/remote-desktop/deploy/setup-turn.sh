#!/usr/bin/env bash
# Install and configure coturn as an EXTERNAL TURN relay for the remote desktop
# app.
#
# g4f-go ships its own embedded STUN/TURN server (`g4f-go turn serve`), which is
# started automatically by `g4f-go -m remote_desktop` and needs no root. Use
# this script only when you want a standalone coturn instance instead, for
# example to share one relay between several hosts.
#
# Run this on the machine that shares its screen:
#
#   sudo bash deploy/setup-turn.sh
#
# It prints the RD_TURN_* environment variables to start the server with.
set -euo pipefail

CONF=/etc/turnserver.conf
SECRET_FILE=/etc/turnserver.secret
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "This script needs root: sudo bash $0" >&2
  exit 1
fi

echo "==> Installing coturn"
if command -v apt-get >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y coturn
elif command -v dnf >/dev/null; then
  dnf install -y coturn
else
  echo "Unsupported package manager; install coturn manually." >&2
  exit 1
fi

echo "==> Detecting the public address"
EXTERNAL_IP="${EXTERNAL_IP:-$(curl -fsS --max-time 10 https://api.ipify.org || true)}"
if [[ -z "${EXTERNAL_IP}" ]]; then
  echo "Could not detect the public IP. Re-run with EXTERNAL_IP=1.2.3.4 sudo -E bash $0" >&2
  exit 1
fi
echo "    external-ip=${EXTERNAL_IP}"

# coturn binds the private address and advertises the public one; without the
# private half it cannot allocate a relay address on a NATed host.
PRIVATE_IP="${PRIVATE_IP:-$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1)}"
if [[ -z "${PRIVATE_IP}" ]]; then
  echo "Could not detect the LAN address. Re-run with PRIVATE_IP=192.168.1.10 sudo -E bash $0" >&2
  exit 1
fi
echo "    private-ip=${PRIVATE_IP}"

echo "==> Generating the shared secret"
if [[ -f "${SECRET_FILE}" ]]; then
  TURN_SECRET="$(cat "${SECRET_FILE}")"
else
  TURN_SECRET="$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 32)"
  printf '%s' "${TURN_SECRET}" > "${SECRET_FILE}"
  chmod 600 "${SECRET_FILE}"
fi

echo "==> Writing ${CONF}"
sed -e "s|__EXTERNAL_IP__|${EXTERNAL_IP}|g" \
    -e "s|__PRIVATE_IP__|${PRIVATE_IP}|g" \
    -e "s|__TURN_SECRET__|${TURN_SECRET}|g" \
    -e "s|__REALM__|${EXTERNAL_IP}|g" \
    "${HERE}/turnserver.conf" > "${CONF}"
chmod 640 "${CONF}"

# Debian/Ubuntu ship coturn disabled until this flag is flipped.
if [[ -f /etc/default/coturn ]]; then
  sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn
fi

echo "==> Opening the firewall"
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 3478/udp
  ufw allow 3478/tcp
  ufw allow 5349/tcp
  ufw allow 49160:49200/udp
elif command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
  firewall-cmd --permanent --add-port=3478/udp
  firewall-cmd --permanent --add-port=3478/tcp
  firewall-cmd --permanent --add-port=5349/tcp
  firewall-cmd --permanent --add-port=49160-49200/udp
  firewall-cmd --reload
else
  echo "    no active ufw/firewalld found, skipping"
fi

echo "==> Starting coturn"
systemctl enable --now coturn
systemctl restart coturn
sleep 1
systemctl --no-pager --lines=5 status coturn || true

cat <<EOF

Done. Start the remote desktop server with:

  RD_TURN_URL="turn:${EXTERNAL_IP}:3478" \\
  RD_TURN_SECRET="${TURN_SECRET}" \\
  python -m remote_desktop --public-url=https://linux.g4f.space

or pass the same values as flags:

  python -m remote_desktop --turn-url "turn:${EXTERNAL_IP}:3478" --turn-secret "${TURN_SECRET}"

Your router must forward these to this machine, or the phone on cellular
still cannot reach the relay:

  UDP 3478, TCP 3478, TCP 5349, UDP 49160-49200

Verify from outside the LAN with https://icetest.info or:

  turnutils_uclient -u user -w pass ${EXTERNAL_IP}
EOF
