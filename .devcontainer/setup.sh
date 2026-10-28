#!/usr/bin/env bash
# Run after the dev container is created and on every start.
set -euo pipefail

# In GitHub Codespaces, /tmp is not what the Docker daemon sees: TIRA has to
# translate its temporary directory to the shared disk, or `tira-cli
# verify-installation` fails with "Docker mounts are not read- and writable".
# CODESPACES=true is set there (and only there). The exports are also written to
# the shell startup files so that every terminal has them.
TIRA_ENV='if [ "${CODESPACES:-}" = "true" ]; then export TIRA_DOCKER_TMPDIR=/mnt/containerTmp; fi'
for rc in ~/.bashrc ~/.zshrc; do
  [ -f "$rc" ] && ! grep -qF TIRA_DOCKER_TMPDIR "$rc" && echo "$TIRA_ENV" >> "$rc"
done
eval "$TIRA_ENV"

uv sync --all-extras
tira-cli verify-installation --local-only
