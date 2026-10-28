FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt source whose signing key has expired, which makes
# every `apt-get update` (e.g. in dev container features) fail. Yarn is not needed.
RUN rm -f /etc/apt/sources.list.d/yarn.list
