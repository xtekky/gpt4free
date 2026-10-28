#!/usr/bin/env bash
#
# start-desktop.sh — start the g4f sandbox desktop container / image.
#
# Launches the TigerVNC + Xfce + noVNC + g4f desktop environment inside Docker.
# The desktop and API server are accessible via browser (noVNC) and direct VNC:
#   Desktop (browser): http://172.0.0.1:8080/desktop/
#   API server:        http://172.0.0.1:8080/
#   VNC direct:        172.0.0.1:5900
#
# The ports are published on SANDBOX_BIND_ADDR (default: 172.0.0.1). Set it to
# 127.0.0.1 for localhost or 0.0.0.0 to expose the desktop on the LAN. The
# container also gets an egress firewall that blocks private, link-local and
# reserved networks; set SANDBOX_BLOCK_LOCAL_NETWORK=0 to disable it.
#
# Usage:
#   ./bin/start-desktop.sh              start in foreground
#   ./bin/start-desktop.sh -d           start in background (detached)
#   ./bin/start-desktop.sh --build      rebuild image before starting
#   ./bin/start-desktop.sh --pull       pull prebuilt image from Docker Hub
#   ./bin/start-desktop.sh --port 8080  use custom port (default: 8080)
#   ./bin/start-desktop.sh --bind 127.0.0.1  use a different host bind address
#   ./bin/start-desktop.sh stop         stop the sandbox container
#   ./bin/start-desktop.sh status       check container status
#   ./bin/start-desktop.sh shell        open interactive bash shell in container
#   ./bin/start-desktop.sh logs         follow container logs
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.sandbox-desktop.yml"
DOCKERFILE="$REPO_ROOT/docker/Dockerfile-sandbox-desktop"

LOCAL_IMAGE="g4f-sandbox:desktop"
REMOTE_IMAGE="hlohaus789/g4f:latest-sandbox-desktop"
IMAGE="${G4F_SANDBOX_IMAGE:-$LOCAL_IMAGE}"
CONTAINER_NAME="g4f-sandbox-desktop"

PORT="${PORT:-8080}"
VNC_PORT="${VNC_PORT:-5900}"
# Host address the published ports bind to. 172.0.0.1 is the default so the
# desktop is not exposed on every interface; override with --bind or
# SANDBOX_BIND_ADDR (127.0.0.1 for localhost, 0.0.0.0 for the LAN).
BIND_ADDR="${SANDBOX_BIND_ADDR:-172.0.0.1}"
BLOCK_LOCAL_NETWORK="${SANDBOX_BLOCK_LOCAL_NETWORK:-1}"
# uid of the unprivileged sandbox user; must match the image's G4F_USER_ID.
USER_ID="${G4F_USER_ID:-1000}"
DETACH=0
BUILD=0
PULL=0
NO_DESKTOP=0
USE_COMPOSE=1
ACTION="up"

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [COMMAND]

Start and manage the g4f sandbox desktop container and image.

Commands:
  up, start       Start the sandbox desktop container (default)
  stop, down      Stop the sandbox desktop container
  restart         Restart the sandbox desktop container
  status          Show status of the sandbox container
  shell, exec     Open a bash shell inside the running container
  logs            View container logs (-f to follow)
  build           Build or rebuild the sandbox desktop image

Options:
  -d, --detach        Run container in the background
  -b, --build         Build/rebuild image before starting
      --pull          Pull prebuilt image from registry ($REMOTE_IMAGE)
  -p, --port PORT     API & noVNC web port (default: $PORT)
      --bind ADDR     Host address to publish the ports on (default: $BIND_ADDR)
      --vnc-port PORT Direct VNC display port (default: $VNC_PORT)
      --user-id UID   uid for the sandbox user (default: $USER_ID)
      --allow-local-network  Let the sandbox reach private/link-local networks
      --no-desktop    Disable desktop GUI stack, run API only (SANDBOX_DESKTOP=0)
      --image IMAGE   Specify Docker image (default: $IMAGE)
      --name NAME     Specify container name (default: $CONTAINER_NAME)
      --no-compose    Use direct 'docker run' instead of 'docker compose'
  -h, --help          Show this help message

Endpoints (published on ${BIND_ADDR}):
  noVNC desktop:  http://${BIND_ADDR}:${PORT}/desktop/
  API server:     http://${BIND_ADDR}:${PORT}/
  Direct VNC:     ${BIND_ADDR}:${VNC_PORT}
EOF
}

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -d|--detach|--background)
            DETACH=1
            shift
            ;;
        -b|--build)
            BUILD=1
            shift
            ;;
        --pull)
            PULL=1
            IMAGE="$REMOTE_IMAGE"
            shift
            ;;
        -p|--port)
            [ $# -ge 2 ] || die "$1 requires an argument"
            PORT="$2"
            shift 2
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --bind|--bind-address)
            [ $# -ge 2 ] || die "$1 requires an argument"
            BIND_ADDR="$2"
            shift 2
            ;;
        --bind=*|--bind-address=*)
            BIND_ADDR="${1#*=}"
            shift
            ;;
        --allow-local-network)
            BLOCK_LOCAL_NETWORK=0
            shift
            ;;
        --vnc-port)
            [ $# -ge 2 ] || die "$1 requires an argument"
            VNC_PORT="$2"
            shift 2
            ;;
        --vnc-port=*)
            VNC_PORT="${1#*=}"
            shift
            ;;
        --user-id|--uid)
            [ $# -ge 2 ] || die "$1 requires an argument"
            USER_ID="$2"
            shift 2
            ;;
        --user-id=*|--uid=*)
            USER_ID="${1#*=}"
            shift
            ;;
        --image)
            [ $# -ge 2 ] || die "$1 requires an argument"
            IMAGE="$2"
            shift 2
            ;;
        --image=*)
            IMAGE="${1#*=}"
            shift
            ;;
        --name)
            [ $# -ge 2 ] || die "$1 requires an argument"
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --name=*)
            CONTAINER_NAME="${1#*=}"
            shift
            ;;
        --no-desktop)
            NO_DESKTOP=1
            shift
            ;;
        --no-compose)
            USE_COMPOSE=0
            shift
            ;;
        up|start)
            ACTION="up"
            shift
            ;;
        stop|down)
            ACTION="stop"
            shift
            ;;
        restart)
            ACTION="restart"
            shift
            ;;
        status)
            ACTION="status"
            shift
            ;;
        shell|exec|bash)
            ACTION="shell"
            shift
            ;;
        logs)
            ACTION="logs"
            shift
            ;;
        build)
            ACTION="build"
            shift
            ;;
        *)
            die "unknown argument: $1 (run $(basename "$0") --help)"
            ;;
    esac
done

case "$USER_ID" in
    ''|*[!0-9]*) die "user id must be numeric: $USER_ID" ;;
esac
[ "$USER_ID" != "0" ] || die "user id 0 (root) is not supported for the sandbox desktop"
[ -n "$BIND_ADDR" ] || die "bind address must not be empty"
case "$BIND_ADDR" in
    *:*) die "IPv6 bind addresses are not supported by this helper; pass it through docker compose instead" ;;
esac
case "$BLOCK_LOCAL_NETWORK" in
    0|1) ;;
    *) die "SANDBOX_BLOCK_LOCAL_NETWORK must be 0 or 1: $BLOCK_LOCAL_NETWORK" ;;
esac

# Find docker CLI
DOCKER_BIN=""
for bin in docker podman; do
    if command -v "$bin" >/dev/null 2>&1; then
        DOCKER_BIN="$bin"
        break
    fi
done

[ -n "$DOCKER_BIN" ] || die "neither 'docker' nor 'podman' found on PATH"

# Check docker compose support
has_compose() {
    if [ "$USE_COMPOSE" -eq 1 ] && [ "$DOCKER_BIN" = "docker" ]; then
        docker compose version >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Action handlers
case "$ACTION" in
    stop)
        log "stopping sandbox desktop container ($CONTAINER_NAME)..."
        if has_compose && [ -f "$COMPOSE_FILE" ]; then
            docker compose -f "$COMPOSE_FILE" down || true
        else
            "$DOCKER_BIN" stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
            "$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        fi
        log "stopped"
        exit 0
        ;;

    status)
        log "checking status of $CONTAINER_NAME..."
        if "$DOCKER_BIN" ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -q "$CONTAINER_NAME"; then
            "$DOCKER_BIN" ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        else
            printf 'container %s is not running\n' "$CONTAINER_NAME"
        fi
        exit 0
        ;;

    shell)
        if ! "$DOCKER_BIN" ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
            die "container $CONTAINER_NAME is not running. Start it first with: $(basename "$0") -d"
        fi
        log "opening bash shell inside $CONTAINER_NAME..."
        exec "$DOCKER_BIN" exec -it -u pn "$CONTAINER_NAME" bash
        ;;

    logs)
        log "showing logs for $CONTAINER_NAME..."
        if has_compose && [ -f "$COMPOSE_FILE" ]; then
            exec docker compose -f "$COMPOSE_FILE" logs -f
        else
            exec "$DOCKER_BIN" logs -f "$CONTAINER_NAME"
        fi
        ;;

    build)
        log "building sandbox desktop image: $IMAGE (user id $USER_ID)..."
        "$DOCKER_BIN" build --build-arg "G4F_USER_ID=$USER_ID" -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT"
        log "build finished: $IMAGE"
        exit 0
        ;;

    restart)
        "$0" stop
        exec "$0" up --user-id "$USER_ID" "$@"
        ;;

    up)
        # Pull requested
        if [ "$PULL" -eq 1 ]; then
            log "pulling $IMAGE..."
            "$DOCKER_BIN" pull "$IMAGE" || die "failed to pull $IMAGE"
            if [ "$USER_ID" != "1000" ]; then
                warn "pulled images run as uid 1000; rebuild locally with --user-id $USER_ID to change it"
            fi
        fi

        # Check if image exists; build if missing or if it was built for a
        # different uid. G4F_USER_ID is recorded in the image environment by the
        # Dockerfile, so an existing image can be checked without running it.
        if [ "$PULL" -eq 0 ] && "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then
            image_user_id="$("$DOCKER_BIN" image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" 2>/dev/null \
                | sed -n 's/^G4F_USER_ID=//p' | tail -n 1)"
            if [ -n "$image_user_id" ] && [ "$image_user_id" != "$USER_ID" ]; then
                log "image '$IMAGE' was built for user id $image_user_id; rebuilding for $USER_ID"
                BUILD=1
            elif [ -z "$image_user_id" ] && [ "$USER_ID" != "1000" ]; then
                log "image '$IMAGE' does not record its user id; rebuilding for $USER_ID"
                BUILD=1
            fi
        fi
        if [ "$BUILD" -eq 1 ] || ! "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then
            if [ "$PULL" -eq 0 ]; then
                log "image '$IMAGE' not found locally (or --build specified); building from $DOCKERFILE..."
                "$DOCKER_BIN" build --build-arg "G4F_USER_ID=$USER_ID" -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT" || die "failed to build $IMAGE"
            fi
        fi

        # Print banner. A 0.0.0.0 publish is reachable through localhost; keep
        # the printed URLs usable instead of showing the wildcard address.
        display_addr="$BIND_ADDR"
        [ "$display_addr" = "0.0.0.0" ] && display_addr="127.0.0.1"
        printf '\n'
        log "Starting G4F Sandbox Desktop..."
        printf '  Container:   %s\n' "$CONTAINER_NAME"
        printf '  Image:       %s\n' "$IMAGE"
        printf '  User ID:     %s\n' "$USER_ID"
        printf '  Bind address: %s\n' "$BIND_ADDR"
        printf '  Desktop URL: http://%s:%s/desktop/\n' "$display_addr" "$PORT"
        printf '  API Server:  http://%s:%s/\n' "$display_addr" "$PORT"
        printf '  App domains: http://g4f.g4f.local/ (also <port>.g4f.local)\n'
        printf '  VNC Display: %s:%s\n' "$display_addr" "$VNC_PORT"
        if [ "$BLOCK_LOCAL_NETWORK" = "1" ]; then
            printf '  Egress:      private/link-local networks blocked\n'
        else
            printf '  Egress:      local networks allowed (SANDBOX_BLOCK_LOCAL_NETWORK=0)\n'
        fi
        printf '\n'

        if [ "$NO_DESKTOP" -eq 1 ]; then
            log "Desktop GUI stack disabled (SANDBOX_DESKTOP=0)"
        fi

        # Run with compose if possible and no custom port/image overrides
        if has_compose && [ -f "$COMPOSE_FILE" ] && [ "$USE_COMPOSE" -eq 1 ] && [ "$PORT" = "8080" ] && [ "$VNC_PORT" = "5900" ] && [ "$IMAGE" = "$LOCAL_IMAGE" ]; then
            COMPOSE_ARGS=("-f" "$COMPOSE_FILE" "up")
            if [ "$DETACH" -eq 1 ]; then
                COMPOSE_ARGS+=("-d")
            fi
            export SANDBOX_DESKTOP="$([ "$NO_DESKTOP" -eq 1 ] && echo 0 || echo 1)"
            export SANDBOX_BIND_ADDR="$BIND_ADDR"
            export SANDBOX_BLOCK_LOCAL_NETWORK="$BLOCK_LOCAL_NETWORK"
            export G4F_USER_ID="$USER_ID"
            exec docker compose "${COMPOSE_ARGS[@]}"
        fi

        # Run with docker run
        # Clean up existing stopped container with the same name
        "$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

        DOCKER_ARGS=(
            run
            --name "$CONTAINER_NAME"
            --hostname sandbox
            --init
            --shm-size 2gb
            --cap-add NET_ADMIN
            -p "${BIND_ADDR}:${PORT}:${PORT}"
            -p "${BIND_ADDR}:${VNC_PORT}:${VNC_PORT}"
            -v "${REPO_ROOT}:/app/gpt4free"
            -e "XDG_RUNTIME_DIR=/tmp/sandbox-runtime"
            -e "SANDBOX_DESKTOP=$([ "$NO_DESKTOP" -eq 1 ] && echo 0 || echo 1)"
            -e "SANDBOX_BLOCK_LOCAL_NETWORK=$BLOCK_LOCAL_NETWORK"
        )

        if [ "$DETACH" -eq 1 ]; then
            DOCKER_ARGS+=("-d")
        else
            DOCKER_ARGS+=("-it" "--rm")
        fi

        DOCKER_ARGS+=("$IMAGE" "g4f-go" "dev" "--port" "$PORT")

        exec "$DOCKER_BIN" "${DOCKER_ARGS[@]}"
        ;;
esac
