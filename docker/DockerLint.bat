docker pull hadolint/hadolint:latest
docker run --rm -i hadolint/hadolint hadolint --ignore DL3007 - < Dockerfile

PAUSE;