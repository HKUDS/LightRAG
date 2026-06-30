#!/usr/bin/env bash
#
# apple-container.sh — Run the LightRAG storage stack on Apple's native
# `container` CLI (https://github.com/apple/container) instead of Docker Compose.
#
# Apple `container` (1.0.0) has no Compose support: no `depends_on`, no
# `healthcheck`, no `condition: service_healthy`, and no `restart` policy. It
# also has no working container-to-container service DNS (the `container system
# dns` domains are not populated with sibling hostnames in 1.0.0; see
# apple/container issue #856). Containers on a shared network DO reach each other
# by IP, however. This script therefore:
#   * recreates Compose ordering with explicit health-wait loops, and
#   * wires each service to its dependencies by the IP that `container` assigns
#     on start (discovered via `container inspect`), so no DNS / sudo is needed.
#
# It mirrors the image tags and the `/proc/net/tcp` health idiom already used by
# the repo's scripts/setup/templates/*.yml.
#
# Stack (all images verified to publish a linux/arm64 manifest):
#   postgres      pgvector/pgvector:pg18                       (KV + doc status)
#   neo4j         neo4j:5-community                            (graph)
#   milvus        milvusdb/milvus:v2.6.11  (standalone, CPU)   (vector)
#   milvus-etcd   quay.io/coreos/etcd:v3.5.25                  (milvus metadata)
#   milvus-minio  minio/minio:RELEASE.2025-09-07T16-13-09Z     (milvus object store)
#   lightrag      ghcr.io/hkuds/lightrag:latest                (API server + WebUI)
#
# LLM + embeddings are reached over normal outbound HTTPS (e.g. OpenAI); no GPU
# and no vLLM services are involved, so this runs on a CPU-only Apple Silicon Mac.
#
# Requirements: macOS 26 (Tahoe) + Apple Silicon + the `container` CLI installed
# and started (`container system start`). Container-to-container networking and
# the `container network` command do not exist before macOS 26, so the stack
# cannot work on older releases.
#
# Usage:
#   scripts/setup/apple-container.sh up [--no-lightrag]
#   scripts/setup/apple-container.sh down [--purge]
#   scripts/setup/apple-container.sh status
#   scripts/setup/apple-container.sh logs <service> [--follow]
#   scripts/setup/apple-container.sh restart <service>
#   scripts/setup/apple-container.sh pull
#   scripts/setup/apple-container.sh help
#
set -euo pipefail

if [[ -z "${BASH_VERSINFO+x}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "Error: scripts/setup/apple-container.sh requires Bash 4 or newer." >&2
  echo "Hint: install a newer bash (e.g. 'brew install bash') and run it via" >&2
  echo "      'bash scripts/setup/apple-container.sh ...'." >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --------------------------------------------------------------------------- #
# Configuration (override via environment before invoking the script)
# --------------------------------------------------------------------------- #
NETWORK="${LIGHTRAG_AC_NETWORK:-lightrag}"
ENV_SOURCE="${LIGHTRAG_AC_ENV_FILE:-$REPO_ROOT/.env}"
ENV_GENERATED="$REPO_ROOT/.apple-container.env"
PLATFORM="linux/arm64"

# Images — keep in sync with scripts/setup/templates/*.yml and docker-compose-full.yml.
# NOTE: postgres uses pgvector/pgvector:pg18 (multi-arch) rather than the
# templates' gzdaniel/postgres-for-rag:pg18-age-pgvector, which is amd64-only and
# has no arm64 manifest. The AGE graph extension it adds is not needed here
# (graph storage is Neo4j, vector storage is Milvus).
IMG_PG="${LIGHTRAG_AC_IMG_PG:-pgvector/pgvector:pg18}"
IMG_NEO4J="${LIGHTRAG_AC_IMG_NEO4J:-neo4j:5-community}"
IMG_MILVUS="${LIGHTRAG_AC_IMG_MILVUS:-milvusdb/milvus:v2.6.11}"
IMG_ETCD="${LIGHTRAG_AC_IMG_ETCD:-quay.io/coreos/etcd:v3.5.25}"
IMG_MINIO="${LIGHTRAG_AC_IMG_MINIO:-minio/minio:RELEASE.2025-09-07T16-13-09Z}"
IMG_LIGHTRAG="${LIGHTRAG_AC_IMG_LIGHTRAG:-ghcr.io/hkuds/lightrag:latest}"

# Memory budgets (per-container VM). Milvus and Neo4j are heavy.
MEM_LIGHT="${LIGHTRAG_AC_MEM_LIGHT:-2G}"
MEM_HEAVY="${LIGHTRAG_AC_MEM_HEAVY:-6G}"

# Credentials / database names (dev defaults; override via environment).
PG_USER="${POSTGRES_USER:-rag}"
PG_PASSWORD="${POSTGRES_PASSWORD:-rag}"
PG_DB="${POSTGRES_DB:-rag}"
NEO4J_USER="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASS="${NEO4J_PASSWORD:-lightragdev}"     # Neo4j 5 requires >= 8 characters
MINIO_USER="${MINIO_ACCESS_KEY_ID:-minioadmin}"
MINIO_PASS="${MINIO_SECRET_ACCESS_KEY:-minioadmin}"

VOLUMES=(rag_pg rag_neo4j rag_milvus rag_etcd rag_minio rag_lightrag)
SERVICES=(postgres neo4j milvus-etcd milvus-minio milvus lightrag)

# Dependency addresses, resolved at `up` time (see cmd_up).
ETCD_ADDR=""; MINIO_ADDR=""; PG_ADDR=""; NEO4J_ADDR=""; MILVUS_ADDR=""

NO_LIGHTRAG="no"
PURGE="no"
LOGS_FOLLOW="no"
DEBUG="${DEBUG:-false}"

# --------------------------------------------------------------------------- #
# Logging (mirrors scripts/setup/setup.sh)
# --------------------------------------------------------------------------- #
COLOR_RESET=""; COLOR_BOLD=""; COLOR_BLUE=""; COLOR_GREEN=""; COLOR_YELLOW=""; COLOR_RED=""

init_colors() {
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    COLOR_RESET=$'\033[0m'; COLOR_BOLD=$'\033[1m'; COLOR_BLUE=$'\033[34m'
    COLOR_GREEN=$'\033[32m'; COLOR_YELLOW=$'\033[33m'; COLOR_RED=$'\033[31m'
  fi
}
log_info()    { echo "${COLOR_BLUE}${COLOR_BOLD}$*${COLOR_RESET}"; }
log_step()    { echo "${COLOR_BLUE}${COLOR_BOLD}$*${COLOR_RESET}"; }
log_warn()    { echo "${COLOR_YELLOW}$*${COLOR_RESET}"; }
log_success() { echo "${COLOR_GREEN}$*${COLOR_RESET}"; }
log_debug()   { [[ "$DEBUG" == "true" ]] && echo "${COLOR_YELLOW}[debug]${COLOR_RESET} $*" || true; }
format_error() {
  echo "${COLOR_RED}${COLOR_BOLD}Error:${COLOR_RESET} ${COLOR_RED}$1${COLOR_RESET}" >&2
  # Use if/then (not &&) so the function always returns 0: a non-zero return
  # here would abort the caller under `set -e` before it can print help / exit.
  if [[ -n "${2:-}" ]]; then echo "  $2" >&2; fi
}

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
preflight() {
  if [[ "$(uname -m)" != "arm64" ]]; then
    format_error "Apple Silicon (arm64) is required." \
      "Apple 'container' runs Linux arm64 VMs; this stack has no amd64 path here."
    exit 1
  fi
  local major
  major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
  if [[ -z "$major" || "$major" -lt 26 ]]; then
    format_error "macOS 26 (Tahoe) or newer is required." \
      "Container-to-container networking and 'container network' do not exist before macOS 26."
    exit 1
  fi
  if ! command -v container >/dev/null 2>&1; then
    format_error "The 'container' CLI was not found." \
      "Install it from https://github.com/apple/container/releases, then run 'container system start'."
    exit 1
  fi
  if ! container system status >/dev/null 2>&1; then
    log_info "Starting container system services..."
    container system start >/dev/null 2>&1 || {
      format_error "'container system start' failed." \
        "Run it manually and accept the default kernel install, then retry."
      exit 1
    }
  fi
}

# --------------------------------------------------------------------------- #
# Idempotent helpers
# --------------------------------------------------------------------------- #
container_exists()  { container ls --all --quiet 2>/dev/null | grep -qx "$1"; }
container_running() { container ls --quiet 2>/dev/null | grep -qx "$1"; }
network_exists()    { container network list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$1"; }
volume_exists()     { container volume list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$1"; }

# container_ip <name> — the IPv4 address `container` assigned on the shared
# network (from `container inspect`, stripping the /24 suffix).
container_ip() {
  container inspect "$1" 2>/dev/null \
    | grep -o '"ipv4Address"[[:space:]]*:[[:space:]]*"[0-9.]*' \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

ensure_network() {
  if network_exists "$NETWORK"; then
    log_debug "network '$NETWORK' already exists"
  else
    log_info "Creating network '$NETWORK'"
    container network create "$NETWORK" >/dev/null
  fi
}

ensure_volumes() {
  local v
  for v in "${VOLUMES[@]}"; do
    if ! volume_exists "$v"; then
      log_debug "creating volume '$v'"
      container volume create "$v" >/dev/null
    fi
  done
}

# wait_for <label> <timeout_s> <cmd...> — poll <cmd> until it succeeds or times out.
wait_for() {
  local label="$1" timeout="$2"; shift 2
  local elapsed=0
  printf '  waiting for %s ' "$label"
  until "$@" >/dev/null 2>&1; do
    if (( elapsed >= timeout )); then
      printf '\n'
      format_error "$label did not become ready within ${timeout}s." \
        "Inspect logs with: ${0##*/} logs ${label}"
      return 1
    fi
    printf '.'
    sleep 2
    elapsed=$(( elapsed + 2 ))
  done
  printf ' ok\n'
  log_success "  $label is ready"
}

# tcp_ready <container> <decimal-port> — true when the port is LISTENing inside
# the container. Mirrors the PORT_HEX /proc/net/tcp idiom from the repo templates;
# needs only a shell + /proc, so it works on every image in the stack.
tcp_ready() {
  local c="$1" port="$2" hex
  hex="$(printf '%04X' "$port")"
  container exec "$c" sh -c "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null | grep -q ':${hex} '"
}

lightrag_health() {
  curl -fsS --max-time 4 "http://127.0.0.1:9621/health" >/dev/null 2>&1
}

# --------------------------------------------------------------------------- #
# Generated env-file for the lightrag container
# --------------------------------------------------------------------------- #
# Copies the user's host .env, then forces the storage backends and connection
# endpoints to the in-network service IPs. The user's .env is never modified (it
# stays host-usable, per the repo's setup-wizard contract).
_set_kv() {
  # _set_kv <file> <key> <value> — drop any existing KEY= line, append the new one.
  local file="$1" key="$2" value="$3"
  grep -vE "^${key}=" "$file" > "${file}.tmp" 2>/dev/null || true
  mv "${file}.tmp" "$file"
  printf '%s=%s\n' "$key" "$value" >> "$file"
}

generate_env_file() {
  if [[ -f "$ENV_SOURCE" ]]; then
    cp "$ENV_SOURCE" "$ENV_GENERATED"
  else
    log_warn "No source .env at $ENV_SOURCE — generating a minimal one (set your LLM keys!)."
    : > "$ENV_GENERATED"
  fi
  # This file is a copy of the user's .env (real API keys) — keep it private.
  chmod 600 "$ENV_GENERATED"

  # Server binds inside the container; the host reaches it via the published port.
  _set_kv "$ENV_GENERATED" HOST "0.0.0.0"
  _set_kv "$ENV_GENERATED" PORT "9621"
  _set_kv "$ENV_GENERATED" WORKING_DIR "/app/data/rag_storage"
  _set_kv "$ENV_GENERATED" INPUT_DIR "/app/data/inputs"

  # Storage backends → external services.
  _set_kv "$ENV_GENERATED" LIGHTRAG_KV_STORAGE "PGKVStorage"
  _set_kv "$ENV_GENERATED" LIGHTRAG_DOC_STATUS_STORAGE "PGDocStatusStorage"
  _set_kv "$ENV_GENERATED" LIGHTRAG_GRAPH_STORAGE "Neo4JStorage"
  _set_kv "$ENV_GENERATED" LIGHTRAG_VECTOR_STORAGE "MilvusVectorDBStorage"

  # Postgres (KV + doc status).
  _set_kv "$ENV_GENERATED" POSTGRES_HOST "$PG_ADDR"
  _set_kv "$ENV_GENERATED" POSTGRES_PORT "5432"
  _set_kv "$ENV_GENERATED" POSTGRES_USER "$PG_USER"
  _set_kv "$ENV_GENERATED" POSTGRES_PASSWORD "$PG_PASSWORD"
  _set_kv "$ENV_GENERATED" POSTGRES_DATABASE "$PG_DB"

  # Neo4j (graph).
  _set_kv "$ENV_GENERATED" NEO4J_URI "neo4j://${NEO4J_ADDR}:7687"
  _set_kv "$ENV_GENERATED" NEO4J_USERNAME "$NEO4J_USER"
  _set_kv "$ENV_GENERATED" NEO4J_PASSWORD "$NEO4J_PASS"

  # Milvus (vector).
  _set_kv "$ENV_GENERATED" MILVUS_URI "http://${MILVUS_ADDR}:19530"

  log_debug "generated env-file at $ENV_GENERATED (PG=$PG_ADDR NEO4J=$NEO4J_ADDR MILVUS=$MILVUS_ADDR)"
}

# --------------------------------------------------------------------------- #
# Service launchers
# --------------------------------------------------------------------------- #
# run_service <name> -- <container run args...> : start if absent, resume if stopped.
run_service() {
  local name="$1"; shift
  if [[ "${1:-}" == "--" ]]; then shift; fi
  if container_running "$name"; then
    log_debug "$name already running"
    return 0
  fi
  if container_exists "$name"; then
    log_info "Starting existing container '$name'"
    container start "$name" >/dev/null
    return 0
  fi
  log_info "Creating container '$name'"
  container run --detach --name "$name" --network "$NETWORK" \
    --platform "$PLATFORM" "$@" >/dev/null
}

start_postgres() {
  run_service postgres -- -m "$MEM_LIGHT" \
    -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASSWORD" -e POSTGRES_DB="$PG_DB" \
    --mount type=volume,source=rag_pg,target=/var/lib/postgresql \
    "$IMG_PG"
}

start_neo4j() {
  run_service neo4j -- -m "$MEM_HEAVY" \
    -e NEO4J_AUTH="${NEO4J_USER}/${NEO4J_PASS}" \
    -e NEO4J_dbms_default__database="neo4j" \
    --mount type=volume,source=rag_neo4j,target=/data \
    "$IMG_NEO4J"
}

start_milvus_deps() {
  run_service milvus-etcd -- \
    -e ETCD_AUTO_COMPACTION_MODE="revision" -e ETCD_AUTO_COMPACTION_RETENTION="1000" \
    -e ETCD_QUOTA_BACKEND_BYTES="4294967296" -e ETCD_SNAPSHOT_COUNT="50000" \
    --mount type=volume,source=rag_etcd,target=/etcd \
    "$IMG_ETCD" \
    etcd -advertise-client-urls=http://0.0.0.0:2379 -listen-client-urls=http://0.0.0.0:2379 -data-dir /etcd

  run_service milvus-minio -- \
    -e MINIO_ROOT_USER="$MINIO_USER" -e MINIO_ROOT_PASSWORD="$MINIO_PASS" \
    --mount type=volume,source=rag_minio,target=/minio_data \
    "$IMG_MINIO" \
    minio server /minio_data --console-address ":9001"
}

start_milvus() {
  run_service milvus -- -m "$MEM_HEAVY" \
    -e ETCD_ENDPOINTS="${ETCD_ADDR}:2379" \
    -e MINIO_ADDRESS="${MINIO_ADDR}:9000" \
    -e MINIO_ACCESS_KEY_ID="$MINIO_USER" -e MINIO_SECRET_ACCESS_KEY="$MINIO_PASS" \
    --mount type=volume,source=rag_milvus,target=/var/lib/milvus \
    "$IMG_MILVUS" \
    milvus run standalone
}

start_lightrag() {
  generate_env_file
  run_service lightrag -- -m "$MEM_LIGHT" \
    --env-file "$ENV_GENERATED" \
    --mount type=volume,source=rag_lightrag,target=/app/data \
    -p 127.0.0.1:9621:9621 \
    "$IMG_LIGHTRAG"
}

# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
cmd_up() {
  preflight
  ensure_network
  ensure_volumes

  log_step "Starting storage services..."
  start_postgres
  start_neo4j
  start_milvus_deps

  wait_for postgres 90 container exec postgres pg_isready -U "$PG_USER" -d "$PG_DB"
  wait_for neo4j 120 tcp_ready neo4j 7687
  wait_for milvus-etcd 60 container exec milvus-etcd etcdctl endpoint health
  wait_for milvus-minio 60 container exec milvus-minio curl -fsS http://localhost:9000/minio/health/live

  # Discover dependency IPs now that etcd/minio are up, then start Milvus.
  ETCD_ADDR="$(container_ip milvus-etcd)"
  MINIO_ADDR="$(container_ip milvus-minio)"
  [[ -n "$ETCD_ADDR" && -n "$MINIO_ADDR" ]] || { format_error "Could not resolve etcd/minio IPs."; exit 1; }
  log_debug "etcd=$ETCD_ADDR minio=$MINIO_ADDR"

  log_step "Starting Milvus standalone..."
  start_milvus
  wait_for milvus 180 tcp_ready milvus 19530

  if [[ "$NO_LIGHTRAG" == "yes" ]]; then
    PG_ADDR="$(container_ip postgres)"; NEO4J_ADDR="$(container_ip neo4j)"; MILVUS_ADDR="$(container_ip milvus)"
    log_success "Storage stack is up. In-network endpoints (set these in your .env):"
    echo "  POSTGRES_HOST=$PG_ADDR  NEO4J_URI=neo4j://${NEO4J_ADDR}:7687  MILVUS_URI=http://${MILVUS_ADDR}:19530"
    echo "  These IPs are reachable from the host too (e.g. psql -h $PG_ADDR -U $PG_USER)."
    return 0
  fi

  # Discover storage IPs and start LightRAG wired to them.
  PG_ADDR="$(container_ip postgres)"
  NEO4J_ADDR="$(container_ip neo4j)"
  MILVUS_ADDR="$(container_ip milvus)"
  [[ -n "$PG_ADDR" && -n "$NEO4J_ADDR" && -n "$MILVUS_ADDR" ]] || { format_error "Could not resolve storage IPs."; exit 1; }

  log_step "Starting LightRAG server..."
  start_lightrag
  wait_for lightrag 120 lightrag_health

  log_success "Stack is up."
  echo "  WebUI : http://127.0.0.1:9621/webui"
  echo "  Health: http://127.0.0.1:9621/health"
  echo "  Neo4j browser : http://${NEO4J_ADDR}:7474   MinIO console: http://${MINIO_ADDR}:9001"
  echo "  (database UIs are served on the container IP, reachable from the host on the vmnet)"
}

cmd_down() {
  preflight
  log_step "Stopping containers..."
  # Tear down in reverse dependency order, derived from SERVICES to avoid drift.
  local svc idx
  for (( idx=${#SERVICES[@]}-1; idx>=0; idx-- )); do
    svc="${SERVICES[idx]}"
    if container_exists "$svc"; then
      container stop "$svc" >/dev/null 2>&1 || true
      container rm "$svc" >/dev/null 2>&1 || true
      log_info "removed $svc"
    fi
  done
  if [[ "$PURGE" == "yes" ]]; then
    log_warn "Purging named volumes (all stored data will be lost)..."
    local v
    for v in "${VOLUMES[@]}"; do
      volume_exists "$v" && container volume delete "$v" >/dev/null 2>&1 || true
    done
    [[ -f "$ENV_GENERATED" ]] && rm -f "$ENV_GENERATED"
  fi
  log_success "Down."
}

cmd_status() {
  preflight
  log_step "Containers"
  container ls --all
  echo
  log_step "Volumes"
  container volume list
  echo
  log_step "Networks"
  container network list
}

cmd_logs() {
  preflight
  local svc="${1:-}"
  [[ -z "$svc" ]] && { format_error "logs requires a service name." "One of: ${SERVICES[*]}"; exit 1; }
  if [[ "$LOGS_FOLLOW" == "yes" ]]; then
    container logs --follow "$svc"
  else
    container logs "$svc"
  fi
}

cmd_restart() {
  preflight
  local svc="${1:-}"
  [[ -z "$svc" ]] && { format_error "restart requires a service name." "One of: ${SERVICES[*]}"; exit 1; }
  container stop "$svc" >/dev/null 2>&1 || true
  container start "$svc" >/dev/null
  log_warn "Restarted $svc. If its IP changed, run '${0##*/} down && ${0##*/} up' to re-wire dependents."
}

cmd_pull() {
  preflight
  local img
  for img in "$IMG_PG" "$IMG_NEO4J" "$IMG_MILVUS" "$IMG_ETCD" "$IMG_MINIO" "$IMG_LIGHTRAG"; do
    log_info "pulling $img ($PLATFORM)"
    container image pull --platform "$PLATFORM" "$img"
  done
  log_success "All images pulled."
}

print_help() {
  local env_src_rel="${ENV_SOURCE#"$REPO_ROOT"/}"
  local env_gen_rel="${ENV_GENERATED#"$REPO_ROOT"/}"
  cat <<EOF
${COLOR_BOLD}apple-container.sh${COLOR_RESET} — LightRAG storage stack on Apple's 'container' CLI

USAGE:
  scripts/setup/apple-container.sh <command> [options]

COMMANDS:
  up [--no-lightrag]     Create network/volumes and start the stack (in order,
                         with health waits). --no-lightrag starts databases only.
  down [--purge]         Stop and remove containers. --purge also deletes volumes
                         (destroys all stored data) and the generated env-file.
  status                 Show containers, volumes and networks.
  logs <service> [--follow]   Show logs for a service.
  restart <service>      Restart one service.
  pull                   Pre-pull all stack images for ${PLATFORM}.
  help                   Show this help.

SERVICES: ${SERVICES[*]}

REQUIREMENTS: macOS 26 (Tahoe), Apple Silicon, and the 'container' CLI (started
with 'container system start'). No sudo and no DNS setup are required: services
are wired by the IP that 'container' assigns on the shared network.

NOTES:
  * Data is stored in named volumes (host bind mounts are unsupported for these
    database images on Apple container). 'down' keeps data unless --purge.
  * Only the LightRAG server publishes a host port (127.0.0.1:9621). The
    databases are NOT published (avoids clashing with a host Postgres on 5432);
    reach them from the host directly on their container IP (see 'status').
  * The user's ${env_src_rel} is never modified; a generated
    ${env_gen_rel} carries the in-network overrides.
EOF
}

# --------------------------------------------------------------------------- #
# Argument parsing / dispatch
# --------------------------------------------------------------------------- #
main() {
  init_colors
  local cmd="${1:-help}"
  shift || true
  # Accept `-h`/`--help` as the first argument, not just the `help` subcommand.
  if [[ "$cmd" == "-h" || "$cmd" == "--help" ]]; then cmd="help"; fi

  local positional=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-lightrag) NO_LIGHTRAG="yes" ;;
      --purge)       PURGE="yes" ;;
      --follow|-f)   LOGS_FOLLOW="yes" ;;
      --debug)       DEBUG="true" ;;
      -h|--help)     cmd="help" ;;
      -*)            format_error "Unknown option: $1"; exit 1 ;;
      *)             positional="$1" ;;
    esac
    shift
  done

  case "$cmd" in
    up)      cmd_up ;;
    down)    cmd_down ;;
    status)  cmd_status ;;
    logs)    cmd_logs "$positional" ;;
    restart) cmd_restart "$positional" ;;
    pull)    cmd_pull ;;
    help|"") print_help ;;
    *)       format_error "Unknown command: $cmd"; print_help; exit 1 ;;
  esac
}

# Only dispatch when executed directly; sourcing (e.g. from tests) just defines
# the functions above.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
