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

RUN echo "Container Tag: ${TF_CONTAINER_VERSION}" && \
    if [ -z "$TL_VERSION" ]; then \
        echo "Building a Nightly Release" \
        && apt-get update \
        && apt-get install -y git \
        && case $TF_CONTAINER_VERSION in \
                latest-py3 | latest-gpu-py3) apt-get install -y python3-tk  ;; \
                *)                           apt-get install -y python-tk ;; \
            esac \
        && mkdir /tensorlayer_dist/ \
        && cd /tensorlayer_dist/ \
        && git clone https://github.com/tensorlayer/tensorlayer.git \
        && cd tensorlayer \
        && pip install --upgrade -e .[all] \
        && apt-get autoremove -y \
        && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Building Tag Release: $TL_VERSION" \
        && apt-get update \
        && case $TF_CONTAINER_VERSION in \
                latest-py3 | latest-gpu-py3) apt-get install -y python3-tk  ;; \
                *)                           apt-get install -y python-tk ;; \
            esac \
        && pip install --upgrade tensorlayer[all]=="$TL_VERSION" \
        && apt-get autoremove -y \
        && rm -rf /var/lib/apt/lists/* ; \
    fi
