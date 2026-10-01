#!/bin/sh
# Starts as root, makes the output volume writable for the user the server runs as, then drops
# to that user (the images volume stays read-only and is never touched).
#
# The user: PUID / PGID when set, otherwise the owner of the mounted output folder (so files
# keep belonging to whoever owns it on the host), or 1000:1000 when that is root (a folder
# Docker created because it did not exist).
set -e
OUT=/data/output

if [ "$(id -u)" != "0" ]; then
    exec "$@"                       # started with --user: nothing to fix, run as given
fi

mkdir -p "$OUT"
owner_uid=$(stat -c %u "$OUT")
owner_gid=$(stat -c %g "$OUT")
if [ -z "$PUID" ]; then
    if [ "$owner_uid" != "0" ]; then PUID=$owner_uid; else PUID=1000; fi
fi
if [ -z "$PGID" ]; then
    if [ "$owner_uid" != "0" ]; then PGID=$owner_gid; else PGID=1000; fi
fi

if [ "$PUID" = "0" ]; then
    mkdir -p "$HOME"
    exec "$@"
fi

as_user() { setpriv --reuid="$PUID" --regid="$PGID" --clear-groups "$@"; }

# only what does not belong to the user yet: a no-op after the first start
if [ "$owner_uid" != "$PUID" ] || [ "$owner_gid" != "$PGID" ]; then
    chown "$PUID:$PGID" "$OUT" 2>/dev/null || true
fi
find "$OUT" \( ! -user "$PUID" -o ! -group "$PGID" \) -exec chown -h "$PUID:$PGID" {} + 2>/dev/null || true

if ! as_user sh -c "test -w '$OUT' && mkdir -p '$HOME'"; then
    echo "ERROR: the output folder (OUTPUT_DIR, mounted at $OUT) is not writable by $PUID:$PGID" >&2
    echo "       and its ownership could not be changed (owner $(stat -c %u:%g "$OUT"), mode $(stat -c %a "$OUT"))." >&2
    echo "       On a NAS share that squashes root, set PUID / PGID in .env to the share's owner," >&2
    echo "       or make the folder writable for $PUID:$PGID on the host." >&2
    exit 1
fi

exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups "$@"
