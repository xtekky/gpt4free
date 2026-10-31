#!/bin/bash
# Egress policy for the sandbox desktop image.
#
# The sandbox must still reach the public internet (the menu installs tools on
# click and the g4f providers talk to their upstream APIs), but it must not be
# able to reach the Docker host's LAN, other containers on the host, or cloud
# metadata endpoints. Those are the addresses an SSRF or a compromised agent
# would use to escape the sandbox, so OUTPUT drops every private, link-local,
# shared (CGNAT) and reserved IPv4 range plus the IPv6 ULA/link-local ranges.
#
# Outbound file-sharing/sync protocols (SMB, NetBIOS, NFS, AFP, FTP, TFTP,
# rsync, iSCSI, rpcbind) and common cryptocurrency-mining pool/node ports are
# rejected as well, so a compromised sandbox cannot exfiltrate files over local
# shares or be turned into a miner. The lists are in the "blocked services"
# section below.
#
# Loopback is intentionally left alone: the g4f API, the domain router, the
# local gateways (Ollama, LiteLLM, LibreChat, ...) and Docker's embedded DNS
# (127.0.0.11) all run inside this network namespace on 127.0.0.1. Only traffic
# that leaves the container is filtered.
#
# Replies to inbound connections are always allowed (ESTABLISHED,RELATED), so
# the published API/noVNC desktop stays reachable from the host even though the
# host/gateway address is itself in a blocked private range. The filter only
# stops the sandbox from OPENING a connection to those ranges.
#
# The rules need CAP_NET_ADMIN. docker-compose.sandbox-desktop.yml and
# bin/start-sandbox-desktop.sh add it for the container; when the image is run
# by hand, pass `--cap-add=NET_ADMIN` as well.
#
# Env knobs:
#   SANDBOX_BLOCK_LOCAL_NETWORK=0  disable the egress filter
#   SANDBOX_FIREWALL_STRICT=1      stop when the filter cannot be applied
#                                  (default: 0, continue with a warning)
#
# Usage:
#   sandbox-firewall.sh            install/replace the filter
#   sandbox-firewall.sh --status   print the installed rules
#   sandbox-firewall.sh --clear    remove the filter

set -euo pipefail

CHAIN4="G4F_SANDBOX_EGRESS"
CHAIN6="G4F_SANDBOX_EGRESS"

# Outbound ports used by file-sharing and remote-sync services. Blocking them
# keeps a compromised sandbox from mounting or pushing files to SMB/CIFS,
# NetBIOS, NFS, AFP, FTP/TFTP, rsync, iSCSI and rpcbind servers.
FILE_SHARING_TCP_PORTS=(
    21    # FTP
    111   # rpcbind/portmapper
    139   # NetBIOS session service
    445   # SMB/CIFS
    548   # Apple Filing Protocol
    873   # rsync
    1214  # Kazaa
    2049  # NFS
    3260  # iSCSI
    4662  # eDonkey/eMule
    6346  # Gnutella
    51413 # Transmission BitTorrent
)
FILE_SHARING_UDP_PORTS=(
    69    # TFTP
    111   # rpcbind/portmapper
    137   # NetBIOS name service
    138   # NetBIOS datagram service
    2049  # NFS
    4672  # eMule
    51413 # Transmission BitTorrent
)
# BitTorrent clients use the 6881-6999 range by default.
FILE_SHARING_TCP_RANGES=(6881:6999)
FILE_SHARING_UDP_RANGES=(6881:6999)

# Destination ports used by common cryptocurrency mining pools and full nodes.
CRYPTO_MINING_TCP_PORTS=(
    3333 3334 3335 3336  # common stratum pools
    4444 5555 7777 9999  # common pool/miner ports
    8332 8333            # Bitcoin RPC / peer-to-peer
    9332 9333            # Litecoin RPC / peer-to-peer
    14444                # common stratum pool
    18080 18081          # Monero peer-to-peer / RPC
    30303                # Ethereum peer-to-peer
    45560 45700          # common mining-pool ports
)
CRYPTO_MINING_UDP_PORTS=(
    3333 3334 3335 3336
    4444 5555 7777 9999
    14444
    18080 18081
    30303
    45560 45700
)

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# The entrypoint runs as `pn`; the image grants that user passwordless sudo and
# the container is started with CAP_NET_ADMIN, so iptables/IPv6 rules can still
# be managed. Running as root works too.
as_root() {
    if [ "$(id -u)" = "0" ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo -n "$@"
    else
        return 1
    fi
}

ipt() {
    as_root iptables -w 5 "$@"
}

ip6t() {
    as_root ip6tables -w 5 "$@"
}

case "${1:-}" in
    -h|--help)
        sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    --clear)
        if ipt -D OUTPUT -j "$CHAIN4" 2>/dev/null; then :; fi
        if ipt -F "$CHAIN4" 2>/dev/null; then :; fi
        if ipt -X "$CHAIN4" 2>/dev/null; then :; fi
        if ip6t -D OUTPUT -j "$CHAIN6" 2>/dev/null; then :; fi
        if ip6t -F "$CHAIN6" 2>/dev/null; then :; fi
        if ip6t -X "$CHAIN6" 2>/dev/null; then :; fi
        log "sandbox egress filter removed"
        exit 0
        ;;
    --status)
        printf 'IPv4 rules:\n'
        ipt -S "$CHAIN4" 2>/dev/null || printf '  (not installed)\n'
        printf 'IPv6 rules:\n'
        ip6t -S "$CHAIN6" 2>/dev/null || printf '  (not installed)\n'
        exit 0
        ;;
    '')
        ;;
    *)
        die "unknown argument: $1 (use --status or --clear)"
        ;;
esac

if [ "${SANDBOX_BLOCK_LOCAL_NETWORK:-1}" = "0" ]; then
    log "sandbox egress filter disabled (SANDBOX_BLOCK_LOCAL_NETWORK=0)"
    exit 0
fi

command -v iptables >/dev/null 2>&1 || die "iptables is not installed"
command -v ip6tables >/dev/null 2>&1 || warn "ip6tables is not installed; IPv6 is not filtered"

# Recreate the chain so a container restart in place picks up the current
# ranges instead of stacking duplicates.
ipt -N "$CHAIN4" 2>/dev/null || true
ipt -F "$CHAIN4" 2>/dev/null || true

# Replies to connections that came IN from outside must be allowed, or the
# published ports (the API/noVNC desktop on 8080, VNC on 5900) are unreachable
# from the host: a host browser connects through docker's DNAT/proxy, so the
# reply's destination is the host/gateway address (a private range below) and
# would otherwise be REJECTed. This opens only established flows -- the sandbox
# still cannot INITIATE a connection to a private or link-local address.
ipt -A "$CHAIN4" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN

# `addrtype --dst-type LOCAL` covers 127.0.0.0/8, the container's own address
# and broadcast, i.e. services in this namespace, without allowing the Docker
# host or any other container.
ipt -A "$CHAIN4" -m addrtype --dst-type LOCAL -j RETURN

# Blocked services: outbound file sharing/sync and crypto mining.
for port in "${FILE_SHARING_TCP_PORTS[@]}"; do
    ipt -A "$CHAIN4" -p tcp --dport "$port" -j REJECT
done
for port in "${FILE_SHARING_UDP_PORTS[@]}"; do
    ipt -A "$CHAIN4" -p udp --dport "$port" -j REJECT
done
for port in "${FILE_SHARING_TCP_RANGES[@]}"; do
    ipt -A "$CHAIN4" -p tcp --dport "$port" -j REJECT
done
for port in "${FILE_SHARING_UDP_RANGES[@]}"; do
    ipt -A "$CHAIN4" -p udp --dport "$port" -j REJECT
done
for port in "${CRYPTO_MINING_TCP_PORTS[@]}"; do
    ipt -A "$CHAIN4" -p tcp --dport "$port" -j REJECT
done
for port in "${CRYPTO_MINING_UDP_PORTS[@]}"; do
    ipt -A "$CHAIN4" -p udp --dport "$port" -j REJECT
done

ipt -A "$CHAIN4" -d 0.0.0.0/8 -j REJECT
ipt -A "$CHAIN4" -d 10.0.0.0/8 -j REJECT
ipt -A "$CHAIN4" -d 100.64.0.0/10 -j REJECT
ipt -A "$CHAIN4" -d 169.254.0.0/16 -j REJECT
ipt -A "$CHAIN4" -d 172.16.0.0/12 -j REJECT
ipt -A "$CHAIN4" -d 192.0.0.0/24 -j REJECT
ipt -A "$CHAIN4" -d 192.0.2.0/24 -j REJECT
ipt -A "$CHAIN4" -d 192.168.0.0/16 -j REJECT
ipt -A "$CHAIN4" -d 198.18.0.0/15 -j REJECT
ipt -A "$CHAIN4" -d 198.51.100.0/24 -j REJECT
ipt -A "$CHAIN4" -d 203.0.113.0/24 -j REJECT
ipt -A "$CHAIN4" -d 224.0.0.0/4 -j REJECT
ipt -A "$CHAIN4" -d 240.0.0.0/4 -j REJECT
ipt -A "$CHAIN4" -d 255.255.255.255/32 -j REJECT

# Hook the chain into OUTPUT once. Keep it first so an existing permissive rule
# cannot shadow the filter.
if ! ipt -C OUTPUT -j "$CHAIN4" >/dev/null 2>&1; then
    ipt -I OUTPUT 1 -j "$CHAIN4"
fi

if command -v ip6tables >/dev/null 2>&1; then
    ip6t -N "$CHAIN6" 2>/dev/null || true
    ip6t -F "$CHAIN6" 2>/dev/null || true
    ip6t -A "$CHAIN6" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
    ip6t -A "$CHAIN6" -m addrtype --dst-type LOCAL -j RETURN
    for port in "${FILE_SHARING_TCP_PORTS[@]}"; do
        ip6t -A "$CHAIN6" -p tcp --dport "$port" -j REJECT
    done
    for port in "${FILE_SHARING_UDP_PORTS[@]}"; do
        ip6t -A "$CHAIN6" -p udp --dport "$port" -j REJECT
    done
    for port in "${FILE_SHARING_TCP_RANGES[@]}"; do
        ip6t -A "$CHAIN6" -p tcp --dport "$port" -j REJECT
    done
    for port in "${FILE_SHARING_UDP_RANGES[@]}"; do
        ip6t -A "$CHAIN6" -p udp --dport "$port" -j REJECT
    done
    for port in "${CRYPTO_MINING_TCP_PORTS[@]}"; do
        ip6t -A "$CHAIN6" -p tcp --dport "$port" -j REJECT
    done
    for port in "${CRYPTO_MINING_UDP_PORTS[@]}"; do
        ip6t -A "$CHAIN6" -p udp --dport "$port" -j REJECT
    done
    ip6t -A "$CHAIN6" -d ::/128 -j REJECT
    ip6t -A "$CHAIN6" -d fc00::/7 -j REJECT
    ip6t -A "$CHAIN6" -d fe80::/10 -j REJECT
    ip6t -A "$CHAIN6" -d ff00::/8 -j REJECT
    ip6t -A "$CHAIN6" -d ::ffff:0:0/96 -j REJECT
    if ! ip6t -C OUTPUT -j "$CHAIN6" >/dev/null 2>&1; then
        ip6t -I OUTPUT 1 -j "$CHAIN6"
    fi
fi

log "sandbox egress filter active (private/link-local/reserved networks, file sharing and crypto mining blocked)"
