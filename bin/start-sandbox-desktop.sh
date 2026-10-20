#!/usr/bin/env bash
#
# start-desktop.sh — start the g4f sandbox desktop container / image.
#
# Launches the TigerVNC + Xfce + noVNC + g4f desktop environment inside Docker.
# The desktop and API server are accessible via browser (noVNC) and direct VNC:
#   Desktop (browser): http://localhost:8080/desktop/
#   API server:        http://localhost:8080/
#   VNC direct:        localhost:5900
#
# Usage:
#   ./bin/start-desktop.sh              start in foreground
#   ./bin/start-desktop.sh -d           start in background (detached)
#   ./bin/start-desktop.sh --build      rebuild image before starting
#   ./bin/start-desktop.sh --pull       pull prebuilt image from Docker Hub
#   ./bin/start-desktop.sh --port 8080  use custom port (default: 8080)
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
      --vnc-port PORT Direct VNC display port (default: $VNC_PORT)
      --no-desktop    Disable desktop GUI stack, run API only (SANDBOX_DESKTOP=0)
      --image IMAGE   Specify Docker image (default: $IMAGE)
      --name NAME     Specify container name (default: $CONTAINER_NAME)
      --no-compose    Use direct 'docker run' instead of 'docker compose'
  -h, --help          Show this help message

Endpoints:
  noVNC desktop:  http://localhost:${PORT}/desktop/
  API server:     http://localhost:${PORT}/
  Direct VNC:     localhost:${VNC_PORT}
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
        --vnc-port)
            [ $# -ge 2 ] || die "$1 requires an argument"
            VNC_PORT="$2"
            shift 2
            ;;
        --vnc-port=*)
            VNC_PORT="${1#*=}"
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
        log "building sandbox desktop image: $IMAGE..."
        "$DOCKER_BIN" build -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT"
        log "build finished: $IMAGE"
        exit 0
        ;;

    restart)
        "$0" stop
        exec "$0" up "$@"
        ;;

    up)
        # Pull requested
        if [ "$PULL" -eq 1 ]; then
            log "pulling $IMAGE..."
            "$DOCKER_BIN" pull "$IMAGE" || die "failed to pull $IMAGE"
        fi

        # Check if image exists; build if missing
        if [ "$BUILD" -eq 1 ] || ! "$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1; then
            if [ "$PULL" -eq 0 ]; then
                log "image '$IMAGE' not found locally (or --build specified); building from $DOCKERFILE..."
                "$DOCKER_BIN" build -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT" || die "failed to build $IMAGE"
            fi
        fi

        # Print banner
        printf '\n'
        log "Starting G4F Sandbox Desktop..."
        printf '  Container:   %s\n' "$CONTAINER_NAME"
        printf '  Image:       %s\n' "$IMAGE"
        printf '  Desktop URL: http://localhost:%s/desktop/\n' "$PORT"
        printf '  API Server:  http://localhost:%s/\n' "$PORT"
        printf '  VNC Display: localhost:%s\n\n' "$VNC_PORT"

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
            -p "${PORT}:${PORT}"
            -p "${VNC_PORT}:${VNC_PORT}"
            -v "${REPO_ROOT}:/app"
            -e "XDG_RUNTIME_DIR=/tmp/sandbox-runtime"
            -e "SANDBOX_DESKTOP=$([ "$NO_DESKTOP" -eq 1 ] && echo 0 || echo 1)"
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
