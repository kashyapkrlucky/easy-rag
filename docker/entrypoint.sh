#!/bin/sh
# Apply database migrations, then hand PID 1 to the server command (CMD).
set -eu

python -m easyrag.storage.migrate

exec "$@"
