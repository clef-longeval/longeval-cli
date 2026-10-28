# Agent instructions

- Docker is the preferred way to run and test code. Do not create Python venvs on the host.
- Use the dev container in `.devcontainer/` (or other existing Dockerfiles/devcontainers), e.g. `devcontainer up --workspace-folder .` then `devcontainer exec --workspace-folder . uv run pytest`.
- Lint with `uv run ruff check` inside the container.
