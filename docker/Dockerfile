# Build args.
#   * Accepted Values:
#        - Python 2 + CPU: "latest"          =>  --build-arg TF_CONTAINER_VERSION="latest"
#        - Python 2 + GPU: "latest-gpu"      =>  --build-arg TF_CONTAINER_VERSION="latest-gpu"
#        - Python 3 + CPU: "latest-py3"      =>  --build-arg TF_CONTAINER_VERSION="latest-py3"
#        - Python 3 + GPU: "latest-gpu-py3"  =>  --build-arg TF_CONTAINER_VERSION="latest-gpu-py3"

ARG TF_CONTAINER_VERSION

FROM tensorflow/tensorflow:${TF_CONTAINER_VERSION}

LABEL version="1.0" maintainer="Jonathan DEKHTIAR <contact@jonathandekhtiar.eu>"

ARG TL_VERSION
ARG TF_CONTAINER_VERSION

RUN echo "Container Tag: ${TF_CONTAINER_VERSION}" \
    && apt-get update \
    && case $TF_CONTAINER_VERSION in \
            latest-py3 | latest-gpu-py3) apt-get install -y python3-tk  ;; \
            *)                           apt-get install -y python-tk ;; \
        esac \
    && if [ -z "$TL_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get install -y git \
        && mkdir /dist/ && cd /dist/ \
        && git clone https://github.com/tensorlayer/tensorlayer.git \
        && cd tensorlayer \
        && pip install --disable-pip-version-check --no-cache-dir --upgrade -e .[all]; \
    else \
        echo "Building Tag Release: $TL_VERSION" \
        && pip install  --disable-pip-version-check --no-cache-dir --upgrade tensorlayer[all]=="$TL_VERSION"; \
    fi \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
