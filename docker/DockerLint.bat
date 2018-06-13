docker pull hadolint/hadolint:latest

docker run --rm -i hadolint/hadolint hadolint --ignore DL3007 - < python2\cpu\Dockerfile
docker run --rm -i hadolint/hadolint hadolint --ignore DL3007 - < python2\gpu\Dockerfile
docker run --rm -i hadolint/hadolint hadolint --ignore DL3007 - < python3\cpu\Dockerfile
docker run --rm -i hadolint/hadolint hadolint --ignore DL3007 - < python3\gpu\Dockerfile

PAUSE;