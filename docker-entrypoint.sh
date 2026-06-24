#!/bin/sh
set -e

# Run the server as the non-root "lightrag" user (CIS Docker 4.1) while staying
# compatible with existing deployments whose bind-mounted data is root-owned.
#
# Image-internal chown cannot fix runtime mounts, so when the container starts
# as root we chown the data dirs and then drop privileges via gosu. When the
# orchestrator already starts us as non-root (compose `user:` / k8s
# `runAsUser`), we skip the chown and exec directly.

# Preserve the pre-split behavior where `docker run <image> --port 9622` appended
# flags to the server. Now that ENTRYPOINT is this script, a first arg starting
# with "-" means the user only passed flags, so prepend the default command.
if [ "${1#-}" != "$1" ]; then
    set -- python -m lightrag.api.lightrag_server "$@"
fi

if [ "$(id -u)" = "0" ]; then
    # Take ownership of the writable data locations so the dropped-privilege
    # process can read/write them, covering bind-mounts/PVCs whose host content
    # is root-owned. We chown /app/data (the default home for all data) plus any
    # custom dirs configured via env (WORKING_DIR/INPUT_DIR/PROMPT_DIR/
    # TIKTOKEN_CACHE_DIR), so deployments that point these outside /app/data
    # keep working. Unset values and system roots are skipped; read-only mounts
    # fail the chown harmlessly.
    for _d in /app/data "$WORKING_DIR" "$INPUT_DIR" "$PROMPT_DIR" "$TIKTOKEN_CACHE_DIR"; do
        case "$_d" in
            ""|/|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/proc|/root|/run|/sbin|/sys|/usr|/var) continue ;;
        esac
        [ -d "$_d" ] && chown -R lightrag:lightrag "$_d" 2>/dev/null || true
    done
    # NOTE: we deliberately do NOT chown /app/.env. On a bind-mount that would
    # change the *host* file's owner to uid 1000, forcing the host user to sudo
    # just to edit their config. .env only needs to be *readable* by uid 1000,
    # which the default 0644 already satisfies. A 0600 .env owned by another uid
    # must be made readable (chmod/chown on the host) or supplied via env vars
    # (compose env_file:/environment:, k8s env/envFrom).
    exec gosu lightrag "$@"
fi

exec "$@"
